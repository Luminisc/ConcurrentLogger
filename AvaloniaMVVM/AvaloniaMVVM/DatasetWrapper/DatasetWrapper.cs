using Avalonia;
using Avalonia.Media.Imaging;
using AvaloniaMVVM.Gpu;
using AvaloniaMVVM.Kernels;
using ILGPU;
using ILGPU.Runtime;
using OSGeo.GDAL;
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace AvaloniaMVVM.DatasetWrapper
{
    public class DatasetWrapper : IDisposable
    {
        public static string picturePath = Path.Combine(Consts.RelativePathToRoot, @"Pics/Data_Envi/samson_1.img");
        //public static string picturePath = Path.Combine(Consts.RelativePathToRoot, @"Pics/moffet_field/f080611t01p00r07rdn_c_sc01_ort_img");
        //public static string picturePath = Path.Combine(Consts.RelativePathToRoot, @"Pics/lowAltitude/f960705t01p02_r02c_img");

        public int width;
        public int height;
        public int depth;

        private readonly Dataset _dataset;

        private MemoryBuffer3D<short> _datasetV;
        private MemoryBuffer3D<byte> _datasetVByte;

        public DatasetWrapper()
        {
            _dataset = Gdal.Open(picturePath, Access.GA_ReadOnly);

            width = _dataset.RasterXSize;
            height = _dataset.RasterYSize;
            depth = _dataset.RasterCount;

            //Width = 20;
            //Height = 20;
            //depth = 100;
        }

        ~DatasetWrapper()
        {
            Dispose();
        }

        public void Dispose()
        {
            _dataset?.Dispose();
            _datasetV?.Dispose();
            _datasetVByte?.Dispose();
            _datasetV = null;
        }

        public void LoadDatasetInVideoMemory()
        {
            int w = width,
                h = height,
                d = depth;

            short[] buffer = new short[w * h * d];
            var bands = Enumerable.Range(1, d).ToArray();
            _dataset.ReadRaster(0, 0, w, h, buffer, w, h, d, bands, 0, 0, 0);
            _datasetV = GpuContext.Instance.Accelerator.Allocate<short>(w, h, d);
            _datasetV.CopyFrom(buffer, 0, Index3.Zero, buffer.Length);
        }

        public void RenderBand(ref WriteableBitmap bmp, int band)
        {
            int w = width, h = height;
            // todo: MinMax should be calculated on GPU
            double[] minMax = new double[2];
            _dataset.GetRasterBand(band).ComputeRasterMinMax(minMax, 0);
            short min = (short)(minMax[0]);
            double mult = 255 / (minMax[1] - minMax[0]);

            uint[] data = CalculationWrappers.PictureConvertion(_datasetV.View, band, mult, min);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
        }

        public void RenderCorrelation(ref WriteableBitmap bmp)
        {
            int w = width, h = height;
            var index = new Index2(w, h);

            uint[] data = CalculationWrappers.GetCorrelationMap(index, _datasetV.View);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
        }

        public void RenderSobel(ref WriteableBitmap bmp)
        {
            int w = width, h = height;
            var index = new Index2(w, h);

            uint[] data = CalculationWrappers.GetSobelMap(index, _datasetV.View);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
        }

        public HistogramData RenderHistogram(ref WriteableBitmap bmp)
        {
            var histData = new HistogramData();
            int w = width, h = height, d = depth;
            var index = new Index3(w, h, d);

            var histIndex = new Index2(1000, 1000);

            var data = CalculationWrappers.GetHistogram(index, _datasetV.View, histIndex, histData);
            bmp = new WriteableBitmap(new Avalonia.PixelSize(histIndex.X, histIndex.Y), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
            return histData;
        }

        public BrightnessCalculationData GetBrightnessCalculationData(ref WriteableBitmap bmp)
        {
            var result = new BrightnessCalculationData();
            result = CalculationWrappers.CalculateBrightnessStats(_datasetV.View, result);

            bmp = new WriteableBitmap(new PixelSize(result.imageSize.X, result.imageSize.Y), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy(result.arrImage, 0, ptr, result.arrImage.Length);
            }

            return result;
        }

        public void ConvertToByteRepresentation(ref WriteableBitmap bmp, short maxValue)
        {
            if (maxValue == 0)
                return;

            var d = _datasetVByte;
            d?.Dispose();
            _datasetVByte = null;
            uint[] data;
            (_datasetVByte, data) = CalculationWrappers.ConvertToByteRepresentation(_datasetV.View, maxValue);

            bmp = new WriteableBitmap(new PixelSize(_datasetVByte.Width, _datasetVByte.Height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888);

            if (data != null)
            {
                using (var buf = bmp.Lock())
                {
                    IntPtr ptr = buf.Address;
                    Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
                }
            }
        }

        public void RenderPearsonCorrelation(ref WriteableBitmap bmp, byte lowThreshold = 0, byte highThreshold = 0)
        {
            var data = CalculationWrappers.CalculatePearsonCorrelation(_datasetVByte, lowThreshold, highThreshold);

            using (var img = new WriteableBitmap(new PixelSize(_datasetVByte.Width, _datasetVByte.Height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888))
            {
                string time = DateTime.Now.ToFileTime().ToString();
                using (var buf = img.Lock())
                {
                    IntPtr ptr = buf.Address;
                    Marshal.Copy((int[])(object)data.xyCorrelation, 0, ptr, data.xyCorrelation.Length);
                }
                img.Save($"{Program.PathToTemp}/PearsonCorrelation_{_datasetVByte.Depth}Bands_{(highThreshold == 0 ? "No" : highThreshold.ToString())}Threshold_{time}.png");

                OpenCvSharp.Mat mt = new OpenCvSharp.Mat(height, width, OpenCvSharp.MatType.CV_8U, data.rawPicture);
                mt.SaveImage($"{Program.PathToTemp}/PearsonCorrelation_{_datasetVByte.Depth}Bands_{(highThreshold == 0 ? "No" : highThreshold.ToString())}Threshold_{time}.bmp");
            }

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data.xyCorrelation, 0, ptr, data.xyCorrelation.Length);
            }


        }

        public void RenderSignatureLengthDerivative(ref WriteableBitmap bmp, bool normalize, short maxValue, byte highThreshold = 0)
        {
            var data = EdgeDetectionWrapper.CalculateSignatureLengthDerivative(_datasetV.View, normalize, maxValue, highThreshold);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data.xyPicture, 0, ptr, data.xyPicture.Length);
            }



            using (var img = new WriteableBitmap(new PixelSize(width, height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888))
            {
                string time = DateTime.Now.ToFileTime().ToString();
                System.Text.StringBuilder sb = new System.Text.StringBuilder();
                sb.Append($"SignatureLengthEdges_short_{depth}Bands_Clamped");
                if (normalize)
                    sb.Append("_normalized");
                sb.Append($"_{time}.png");

                using (var buf = img.Lock())
                {
                    IntPtr ptr = buf.Address;
                    Marshal.Copy((int[])(object)data.xyPicture, 0, ptr, data.xyPicture.Length);
                }
                img.Save($"{Program.PathToTemp}/{sb.ToString()}");

                OpenCvSharp.Mat mt = new OpenCvSharp.Mat(height, width, OpenCvSharp.MatType.CV_8U, data.rawPicture);
                mt.SaveImage($"{Program.PathToTemp}/{sb.ToString()}.bmp");
            }
        }

        public void RenderByteSignatureLengthDerivative(ref WriteableBitmap bmp, bool normalize)
        {
            var data = EdgeDetectionWrapper.CalculateSignatureLengthDerivative(_datasetVByte.View, normalize);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data.xyPicture, 0, ptr, data.xyPicture.Length);
            }



            using (var img = new WriteableBitmap(new PixelSize(width, height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888))
            {
                string time = DateTime.Now.ToFileTime().ToString();
                //using (var buf = img.Lock())
                //{
                //    IntPtr ptr = buf.Address;
                //    Marshal.Copy((int[])(object)data.xPicture, 0, ptr, data.xPicture.Length);
                //}
                //img.Save($"{Program.PathToTemp}/xPicture_{time}.png");
                //using (var buf = img.Lock())
                //{
                //    IntPtr ptr = buf.Address;
                //    Marshal.Copy((int[])(object)data.yPicture, 0, ptr, data.yPicture.Length);
                //}
                //img.Save($"{Program.PathToTemp}/yPicture_{time}.png");
                System.Text.StringBuilder sb = new System.Text.StringBuilder();
                sb.Append($"SignatureLengthEdges_byte_{depth}Bands");
                if (normalize)
                    sb.Append("_normalized");
                sb.Append($"_{time}.png");

                using (var buf = img.Lock())
                {
                    IntPtr ptr = buf.Address;
                    Marshal.Copy((int[])(object)data.xyPicture, 0, ptr, data.xyPicture.Length);
                }
                img.Save($"{Program.PathToTemp}/{sb.ToString()}");

                OpenCvSharp.Mat mt = new OpenCvSharp.Mat(height, width, OpenCvSharp.MatType.CV_8U, data.rawPicture);
                mt.SaveImage($"{Program.PathToTemp}/{sb.ToString()}.bmp");
            }
        }

        public void AccumulateEdges(ref WriteableBitmap bmp, byte[] cannyData, byte pearsonThreshold, short maxValue)
        {
            uint[] data = EdgeDetectionWrapper.AccumulateEdges(_datasetV, _datasetVByte, cannyData, pearsonThreshold, maxValue);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
        }

        public void RenderPseudoColor(ref WriteableBitmap bmp, short max = -1)
        {
            var redBand = 11 - 1;
            var greenBand = 17 - 1;
            var blueBand = 38 - 1;

            double[] minMax = new double[2];
            _dataset.GetRasterBand(redBand).ComputeRasterMinMax(minMax, 0);
            short redMax = (short)(minMax[1]);
            _dataset.GetRasterBand(greenBand).ComputeRasterMinMax(minMax, 0);
            short greenMax = (short)(minMax[1]);
            _dataset.GetRasterBand(blueBand).ComputeRasterMinMax(minMax, 0);
            short blueMax = (short)(minMax[1]);
            if (max == -1)
                max = XMath.Max(blueMax, Math.Max(redMax, greenMax));

            uint[] data = CalculationWrappers.CalculatePseudoColor(_datasetV.View, redBand, greenBand, blueBand, max);

            bmp = new WriteableBitmap(new PixelSize(width, height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
        }

        public void RenderScanline(ref WriteableBitmap bmp, int band, int row, int column)
        {
            var data = CalculationWrappers.CalcScanlineImage(_datasetV.View, band, row, column);

            bmp = new WriteableBitmap(new Avalonia.PixelSize(width + depth, height + depth), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
        }
    }
}
