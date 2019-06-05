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
        //public static string picturePath = Path.Combine(Consts.relativePathToRoot, @"Pics/Data_Envi/samson_1.img");
        public static string picturePath = Path.Combine(Consts.relativePathToRoot, @"Pics/moffet_field/f080611t01p00r07rdn_c_sc01_ort_img");

        public int Width;
        public int Height;
        public int Depth;

        private Dataset dataset;

        private MemoryBuffer3D<short> dataset_v;
        private MemoryBuffer3D<byte> dataset_v_byte;

        public DatasetWrapper()
        {
            dataset = Gdal.Open(picturePath, Access.GA_ReadOnly);

            Width = dataset.RasterXSize;
            Height = dataset.RasterYSize;
            Depth = dataset.RasterCount;

            //Width = 20;
            //Height = 20;
            //Depth = 100;
        }

        ~DatasetWrapper()
        {
            Dispose();
        }

        public void Dispose()
        {
            dataset?.Dispose();
            dataset_v?.Dispose();
            dataset_v_byte?.Dispose();
            if (dataset_v != null) dataset_v = null;
        }

        public void LoadDatasetInVideoMemory()
        {
            int w = Width,
                h = Height,
                d = Depth;

            short[] buffer = new short[w * h * d];
            var bands = Enumerable.Range(1, d).ToArray();
            dataset.ReadRaster(0, 0, w, h, buffer, w, h, d, bands, 0, 0, 0);
            dataset_v = GpuContext.Instance.Accelerator.Allocate<short>(w, h, d);
            dataset_v.CopyFrom(buffer, 0, Index3.Zero, buffer.Length);
        }

        public void RenderBand(ref WriteableBitmap bmp, int band)
        {
            int w = Width, h = Height;
            // todo: MinMax should be calculated on GPU
            double[] MinMax = new double[2];
            dataset.GetRasterBand(band).ComputeRasterMinMax(MinMax, 0);
            short min = (short)(MinMax[0]);
            double mult = 255 / (MinMax[1] - MinMax[0]);

            uint[] data = CalculationWrappers.PicConvertion(w * h, dataset_v.View, band, mult, min);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
        }

        public void RenderCorrelation(ref WriteableBitmap bmp)
        {
            int w = Width, h = Height;
            var index = new Index2(w, h);

            uint[] data = CalculationWrappers.GetCorrelationMap(index, dataset_v.View);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
        }

        public void RenderSobel(ref WriteableBitmap bmp)
        {
            int w = Width, h = Height;
            var index = new Index2(w, h);

            uint[] data = CalculationWrappers.GetSobelMap(index, dataset_v.View);

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
        }

        public HistogramData RenderHistogram(ref WriteableBitmap bmp)
        {
            var histData = new HistogramData();
            int w = Width, h = Height, d = Depth;
            var index = new Index3(w, h, d);

            var histIndex = new Index2(1000, 1000);

            var data = CalculationWrappers.GetHistogram(index, dataset_v.View, histIndex, histData);
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
            result = CalculationWrappers.CalculateBrightnessStats(dataset_v.View, result);

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

            var d = dataset_v_byte;
            d?.Dispose();
            dataset_v_byte = null;
            uint[] data;
            (dataset_v_byte, data) = CalculationWrappers.ConvertToByteRepresentation(dataset_v.View, maxValue);

            bmp = new WriteableBitmap(new PixelSize(dataset_v_byte.Width, dataset_v_byte.Height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888);

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
            var data = CalculationWrappers.CalculatePearsonCorrelation(dataset_v_byte, lowThreshold, highThreshold);

            if (highThreshold == 0 || lowThreshold == 0)
            {
                using (var img = new WriteableBitmap(new PixelSize(dataset_v_byte.Width, dataset_v_byte.Height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888))
                {
                    string time = DateTime.Now.ToFileTime().ToString();
                    using (var buf = img.Lock())
                    {
                        IntPtr ptr = buf.Address;
                        Marshal.Copy((int[])(object)data.xCorrelation, 0, ptr, data.xCorrelation.Length);
                    }
                    img.Save($"D://Temp/xCorrelation_{time}.png");
                    using (var buf = img.Lock())
                    {
                        IntPtr ptr = buf.Address;
                        Marshal.Copy((int[])(object)data.yCorrelation, 0, ptr, data.yCorrelation.Length);
                    }
                    img.Save($"D://Temp/yCorrelation_{time}.png");
                    using (var buf = img.Lock())
                    {
                        IntPtr ptr = buf.Address;
                        Marshal.Copy((int[])(object)data.xyCorrelation, 0, ptr, data.xyCorrelation.Length);
                    }
                    img.Save($"D://Temp/xyCorrelation_{time}.png");
                }
            }
            
            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data.xyCorrelation, 0, ptr, data.xyCorrelation.Length);
            }
        }
    }
}
