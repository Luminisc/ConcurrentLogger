using AvaloniaMVVM.Gpu;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AvaloniaMVVM.Kernels
{
    public static class CalculationWrappers
    {
        static Action<Index, ArrayView<uint>, ArrayView<short>, double, short> pinConvertKernel { get { return GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<uint>, ArrayView<short>, double, short>(Kernels.PicConvertion); } }
        static Action<Index, ArrayView<uint>, ArrayView<byte>, double, short> pinConvertFromByteKernel { get { return GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<uint>, ArrayView<byte>, double, short>(Kernels.PicConvertionByte); } }
        static Action<Index, ArrayView<short>, ArrayView<short>> noOpKernel { get { return GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<short>, ArrayView<short>>(Kernels.NoOp); } }

        public static uint[] PicConvertion(Index index, ArrayView3D<short> imageView, int band, double mult, short min)
        {
            uint[] result = null;
            using (var bufOut = GpuContext.Instance.Accelerator.Allocate<uint>(imageView.Width * imageView.Height))
            {
                pinConvertKernel(bufOut.Length, bufOut.View, imageView.GetSliceView(band - 1).AsLinearView(), mult, min);
                GpuContext.Instance.Accelerator.Synchronize();
                result = bufOut.GetAsArray();
            }

            return result;
        }

        public static uint[] PicConvertion(Index index, ArrayView3D<byte> imageView, int band, double mult, short min)
        {
            uint[] result = null;
            using (var bufOut = GpuContext.Instance.Accelerator.Allocate<uint>(imageView.Width * imageView.Height))
            {
                pinConvertFromByteKernel(bufOut.Length, bufOut.View, imageView.GetSliceView(band - 1).AsLinearView(), mult, min);
                GpuContext.Instance.Accelerator.Synchronize();
                result = bufOut.GetAsArray();
            }

            return result;
        }

        #region Correlation


        public static uint[] GetCorrelationMap(Index2 index, ArrayView3D<short> imageView)
        {
            var horizontalCorrelationMapKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<double>, ArrayView3D<short>>(Kernels.HorizontalCorrelationMap);
            var verticalCorrelationMapKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<double>, ArrayView3D<short>>(Kernels.VerticalCorrelationMap);
            var dlbCorrelationMapKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<double>, ArrayView3D<short>>(Kernels.DiagLeftBottomCorrelationMap);
            var drbCorrelationMapKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<double>, ArrayView3D<short>>(Kernels.DiagRightBottomCorrelationMap);
            var buildCorrelationMapKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<double>, ArrayView2D<double>, ArrayView2D<double>, ArrayView2D<double>, ArrayView2D<double>>(Kernels.BuildCorrelationMap);
            var correlationMapToRgbaKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<uint>, ArrayView2D<double>>(Kernels.CorrelationMapToRgba32);

            uint[] result = null;

            using (var bufHor = GpuContext.Instance.Accelerator.Allocate<double>(index))
            using (var bufVer = GpuContext.Instance.Accelerator.Allocate<double>(index))
            using (var bufDlb = GpuContext.Instance.Accelerator.Allocate<double>(index))
            using (var bufDrb = GpuContext.Instance.Accelerator.Allocate<double>(index))
            using (var bufBor = GpuContext.Instance.Accelerator.Allocate<double>(index))
            using (var bufOut2 = GpuContext.Instance.Accelerator.Allocate<uint>(index))
            {
                horizontalCorrelationMapKernel(index, bufHor.View, imageView);
                GpuContext.Instance.Accelerator.Synchronize();
                verticalCorrelationMapKernel(index, bufVer.View, imageView);
                GpuContext.Instance.Accelerator.Synchronize();
                dlbCorrelationMapKernel(index, bufDlb.View, imageView);
                GpuContext.Instance.Accelerator.Synchronize();
                drbCorrelationMapKernel(index, bufDrb.View, imageView);
                GpuContext.Instance.Accelerator.Synchronize();

                buildCorrelationMapKernel(index, bufBor.View, bufHor.View, bufVer.View, bufDlb.View, bufDrb.View);
                GpuContext.Instance.Accelerator.Synchronize();

                correlationMapToRgbaKernel(index, bufOut2.View, bufBor.View);
                GpuContext.Instance.Accelerator.Synchronize();

                result = bufOut2.GetAsArray();
            }

            return result;
        }
        #endregion

        #region Sobel

        public static uint[] GetSobelMap(Index2 index, ArrayView3D<short> imageView)
        {
            var horizontalSobelKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView3D<short>, ArrayView3D<short>>(Kernels.CalculateHorizontalSobel);
            var verticalSobelKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView3D<short>, ArrayView3D<short>>(Kernels.CalculateVerticalSobel);
            var accumulateSobelKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index3, ArrayView3D<short>, ArrayView3D<short>, ArrayView3D<short>>(Kernels.AccumulateSobelMap);
            var calculateSobelKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<short>, ArrayView3D<short>>(Kernels.CalculateSobel);

            uint[] result = null;

            using (var bufHor = GpuContext.Instance.Accelerator.Allocate<short>(imageView.Extent))
            using (var bufVer = GpuContext.Instance.Accelerator.Allocate<short>(imageView.Extent))
            using (var bufAccum = GpuContext.Instance.Accelerator.Allocate<short>(imageView.Extent))
            using (var bufMap = GpuContext.Instance.Accelerator.Allocate<short>(index))
            using (var bufOut2 = GpuContext.Instance.Accelerator.Allocate<uint>(index))
            {
                horizontalSobelKernel(index, bufHor.View, imageView);
                GpuContext.Instance.Accelerator.Synchronize();
                verticalSobelKernel(index, bufVer.View, imageView);
                GpuContext.Instance.Accelerator.Synchronize();
                accumulateSobelKernel(bufAccum.Extent, bufAccum.View, bufHor.View, bufVer.View);
                GpuContext.Instance.Accelerator.Synchronize();
                calculateSobelKernel(index, bufMap.View, bufAccum.View);
                GpuContext.Instance.Accelerator.Synchronize();

                pinConvertKernel(index.Size, bufOut2.AsLinearView(), bufMap.AsLinearView(), 1, 0);
                GpuContext.Instance.Accelerator.Synchronize();

                result = bufOut2.GetAsArray();
            }

            return result;
        }

        #endregion

        public static uint[] GetHistogram(Index3 index, ArrayView3D<short> imageView, Index2 histImageIndex, HistogramData histData)
        {
            var histogramKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index3, ArrayView<int>, ArrayView3D<short>>(HistogramKernels.CalculateHistogram);
            var histogramImageKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView2D<short>, ArrayView<int>, double, double, int>(HistogramKernels.CalculateHistogramImage);

            uint[] result = null;
            int[] buf1;

            using (var bufHist = GpuContext.Instance.Accelerator.Allocate<int>(short.MaxValue))
            {
                bufHist.MemSetToZero();
                histogramKernel(index, bufHist.View, imageView);
                GpuContext.Instance.Accelerator.Synchronize();

                buf1 = bufHist.GetAsArray();
                var lastElement = KernelHelpers.FindLastNotZeroIndex(buf1);
                histData.histogramData = buf1.Take(lastElement + 1).ToArray();
                var max = buf1.Skip(1).Max();


                using (var bufHistImage = GpuContext.Instance.Accelerator.Allocate<short>(histImageIndex))
                using (var bufOut2 = GpuContext.Instance.Accelerator.Allocate<uint>(histImageIndex.Size))
                {
                    bufHistImage.MemSetToZero();
                    histogramImageKernel(new Index(lastElement + 1), bufHistImage.View, bufHist.View.GetSubView(0, lastElement + 1), 1000, 1000, max);
                    GpuContext.Instance.Accelerator.Synchronize();

                    pinConvertKernel(histImageIndex.Size, bufOut2, bufHistImage.AsLinearView(), 1, 0);
                    GpuContext.Instance.Accelerator.Synchronize();

                    result = bufOut2.GetAsArray();
                }

            }

            return result;
        }

        public static BrightnessCalculationData CalculateBrightnessStats(ArrayView3D<short> imageView, BrightnessCalculationData output)
        {
            var calcStatsKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView3D<short>, ArrayView<double>, ArrayView<short>, ArrayView<double>>(BrightnessKernels.CalculateBrightnessStats);
            var memSetKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<int>, int>(Kernels.Memset);
            var calcMapKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<double>, ArrayView<short>, ArrayView<double>, ArrayView2D<int>, int, int, int, double>(BrightnessKernels.CalculateBrightnessMap);


            using (var MeanBrightnessBuf = GpuContext.Instance.Accelerator.Allocate<double>(imageView.Depth))
            using (var MaxBrightnessBuf = GpuContext.Instance.Accelerator.Allocate<short>(imageView.Depth))
            using (var DeviationBuf = GpuContext.Instance.Accelerator.Allocate<double>(imageView.Depth))

            {
                calcStatsKernel(imageView.Depth, imageView, MeanBrightnessBuf.View, MaxBrightnessBuf.View, DeviationBuf.View);
                GpuContext.Instance.Accelerator.Synchronize();

                var max = MaxBrightnessBuf.GetAsArray().Max();
                var dMax = (int)DeviationBuf.GetAsArray().Max();
                var height = Math.Max(max, dMax) + 1;
                output.imageSize = new Index2(1000, 1000);
                double scale = 1000 / (double)height;

                using (var ImageBuf = GpuContext.Instance.Accelerator.Allocate<int>(output.imageSize))
                {
                    memSetKernel(output.imageSize.Size, ImageBuf.AsLinearView(), ColorConsts.Black);
                    GpuContext.Instance.Accelerator.Synchronize();

                    calcMapKernel(imageView.Depth, MeanBrightnessBuf.View, MaxBrightnessBuf.View, DeviationBuf.View, ImageBuf.View, ColorConsts.Red, ColorConsts.Blue, ColorConsts.Green, scale);
                    GpuContext.Instance.Accelerator.Synchronize();

                    output.arrImage = ImageBuf.GetAsArray();
                    output.arrMeanBrightness = MeanBrightnessBuf.GetAsArray();
                    output.arrMaxBrightness = MaxBrightnessBuf.GetAsArray();
                    output.arrStandartDeviation = DeviationBuf.GetAsArray();
                }
            }

            return output;
        }

        public static (MemoryBuffer3D<byte>, uint[]) ConvertToByteRepresentation(ArrayView3D<short> imageView, short maxValue)
        {
            var outputBuf = GpuContext.Instance.Accelerator.Allocate<byte>(imageView.Extent);
            var convertToByteKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index3, ArrayView3D<byte>, ArrayView3D<short>, short>(Kernels.ConvertToByte);

            convertToByteKernel(imageView.Extent, outputBuf.View, imageView, maxValue);
            GpuContext.Instance.Accelerator.Synchronize();
            uint[] buff;

            var slice = outputBuf.GetSliceView(30);

            using (var ImageBuf = GpuContext.Instance.Accelerator.Allocate<uint>(slice.Extent))
            {
                pinConvertFromByteKernel(imageView.GetSliceView(0).AsLinearView().Extent, ImageBuf.AsLinearView(), slice.AsLinearView(), 1.0, 0);
                GpuContext.Instance.Accelerator.Synchronize();

                buff = ImageBuf.GetAsArray();
            }

            return (outputBuf, buff);
        }

        public static CorrelationData CalculatePearsonCorrelation(ArrayView3D<byte> imageView, byte lowThreshold = 0, byte highThreshold = 0)
        {
            var pearsonKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView3D<float>, ArrayView3D<byte>>(CorrelationKernels.CorrelationMap);
            var normilizeKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<float>, float>(Kernels.Normalize);
            var byteToFloatKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<float>, ArrayView<byte>>(Kernels.FloatToByte);
            var result = new CorrelationData();

            var slice = imageView.GetSliceView(0);
            var comboSlice = new Index3(slice.Width, slice.Height, 3);

            using (var calcBuf = GpuContext.Instance.Accelerator.Allocate<float>(comboSlice))
            using (var byteBuf = GpuContext.Instance.Accelerator.Allocate<byte>(comboSlice))
            using (var ImageBuf = GpuContext.Instance.Accelerator.Allocate<uint>(comboSlice))
            {
                pearsonKernel(slice.Extent, calcBuf.View, imageView);
                GpuContext.Instance.Accelerator.Synchronize();

                var maxCorrelation = calcBuf.GetAsArray().Max();
                normilizeKernel(calcBuf.Extent.Size, calcBuf.AsLinearView(), maxCorrelation);
                GpuContext.Instance.Accelerator.Synchronize();

                var maxGradientValue = calcBuf.GetAsArray().Skip(slice.Extent.Size).Take(slice.Extent.Size).Max();
                normilizeKernel(slice.Extent.Size, calcBuf.GetSliceView(1).AsLinearView(), maxGradientValue);
                GpuContext.Instance.Accelerator.Synchronize();

                byteToFloatKernel(calcBuf.Extent.Size, calcBuf.AsLinearView(), byteBuf.AsLinearView());
                GpuContext.Instance.Accelerator.Synchronize();

                pinConvertFromByteKernel(slice.Extent.Size, ImageBuf.GetSliceView(0).AsLinearView(), byteBuf.GetSliceView(0).AsLinearView(), 1.0, 0);
                pinConvertFromByteKernel(slice.Extent.Size, ImageBuf.GetSliceView(1).AsLinearView(), byteBuf.GetSliceView(1).AsLinearView(), 1.0, 0);
                pinConvertFromByteKernel(slice.Extent.Size, ImageBuf.GetSliceView(2).AsLinearView(), byteBuf.GetSliceView(2).AsLinearView(), 1.0, 0);
                GpuContext.Instance.Accelerator.Synchronize();

                var img = ImageBuf.GetAsArray();
                var size = slice.Extent.Size;

                var buf = new uint[size];
                Array.Copy(img, 0, buf, 0, size);
                result.xCorrelation = buf;

                buf = new uint[size];
                Array.Copy(img, size, buf, 0, size);
                result.yCorrelation = buf;

                buf = new uint[size];
                Array.Copy(img, size * 2, buf, 0, size);
                result.xyCorrelation = buf;

            }

            return result;
        }
    }
}
