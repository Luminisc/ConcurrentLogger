using AvaloniaMVVM.Gpu;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Linq;

namespace AvaloniaMVVM.Kernels
{
    public static class CalculationWrappers
    {
        private static readonly Action<Index, ArrayView<uint>, ArrayView<short>, double, short> PiсConvertKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<uint>, ArrayView<short>, double, short>(Kernels.PicConvertion);

        private static readonly Action<Index, ArrayView<uint>, ArrayView<byte>, double, short> PiсConvertFromByteKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<uint>, ArrayView<byte>, double, short>(Kernels.PicConvertionByte);

        public static uint[] PictureConvertion(ArrayView3D<short> imageView, int band, double mult, short min)
        {
            uint[] result;
            using (var bufOut = GpuContext.Instance.Accelerator.Allocate<uint>(imageView.Width * imageView.Height))
            {
                PiсConvertKernel(bufOut.Length, bufOut.View, imageView.GetSliceView(band - 1).AsLinearView(), mult, min);
                GpuContext.Instance.Accelerator.Synchronize();
                result = bufOut.GetAsArray();
            }

            return result;
        }

        public static uint[] PictureConvertion(ArrayView3D<byte> imageView, int band, double mult, short min)
        {
            uint[] result;
            using (var bufOut = GpuContext.Instance.Accelerator.Allocate<uint>(imageView.Width * imageView.Height))
            {
                PiсConvertFromByteKernel(bufOut.Length, bufOut.View, imageView.GetSliceView(band - 1).AsLinearView(), mult, min);
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

            uint[] result;

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

            uint[] result;

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

                PiсConvertKernel(index.Size, bufOut2.AsLinearView(), bufMap.AsLinearView(), 1, 0);
                GpuContext.Instance.Accelerator.Synchronize();

                result = bufOut2.GetAsArray();
            }

            return result;
        }

        #endregion

        public static uint[] GetHistogram(Index3 index, ArrayView3D<short> imageView, /*remove*/Index2 histImageIndex, HistogramData histData)
        {
            var histogramKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index3, ArrayView<int>, ArrayView3D<short>>(HistogramKernels.CalculateHistogram);
            var histogramImageKernel = GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView2D<short>, ArrayView<int>, double, double, int>(HistogramKernels.CalculateHistogramImage);

            uint[] result;
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

                    PiсConvertKernel(histImageIndex.Size, bufOut2, bufHistImage.AsLinearView(), 1, 0);
                    GpuContext.Instance.Accelerator.Synchronize();

                    result = bufOut2.GetAsArray();
                }

            }

            return result;
        }

        private static readonly Action<Index, ArrayView3D<short>, ArrayView<double>, ArrayView<short>, ArrayView<double>> _calcStatsKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView3D<short>, ArrayView<double>, ArrayView<short>, ArrayView<double>>(BrightnessKernels.CalculateBrightnessStats);

        private static readonly Action<Index, ArrayView<int>, int> _memSetKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<int>, int>(Kernels.Memset);

        private static readonly Action<Index, ArrayView<double>, ArrayView<short>, ArrayView<double>, ArrayView2D<int>, int, int, int, double> _calcMapKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<double>, ArrayView<short>, ArrayView<double>, ArrayView2D<int>, int, int, int, double>(BrightnessKernels.CalculateBrightnessMap);

        public static BrightnessCalculationData CalculateBrightnessStats(ArrayView3D<short> imageView, BrightnessCalculationData output)
        {
            using (var meanBrightnessBuf = GpuContext.Instance.Accelerator.Allocate<double>(imageView.Depth))
            using (var maxBrightnessBuf = GpuContext.Instance.Accelerator.Allocate<short>(imageView.Depth))
            using (var deviationBuf = GpuContext.Instance.Accelerator.Allocate<double>(imageView.Depth))
            {
                _calcStatsKernel(imageView.Depth, imageView, meanBrightnessBuf.View, maxBrightnessBuf.View, deviationBuf.View);
                GpuContext.Instance.Accelerator.Synchronize();

                var max = maxBrightnessBuf.GetAsArray().Max();
                var dMax = (int)deviationBuf.GetAsArray().Max();
                var height = Math.Max(max, dMax) + 1;
                output.imageSize = new Index2(1000, 1000);
                double scale = 1000 / (double)height;

                using (var imageBuf = GpuContext.Instance.Accelerator.Allocate<int>(output.imageSize))
                {
                    _memSetKernel(output.imageSize.Size, imageBuf.AsLinearView(), ColorConsts.Black);
                    GpuContext.Instance.Accelerator.Synchronize();

                    _calcMapKernel(imageView.Depth, meanBrightnessBuf.View, maxBrightnessBuf.View, deviationBuf.View, imageBuf.View, ColorConsts.Red, ColorConsts.Blue, ColorConsts.Green, scale);
                    GpuContext.Instance.Accelerator.Synchronize();

                    output.arrImage = imageBuf.GetAsArray();
                    output.arrMeanBrightness = meanBrightnessBuf.GetAsArray();
                    output.arrMaxBrightness = maxBrightnessBuf.GetAsArray();
                    output.arrStandardDeviation = deviationBuf.GetAsArray();
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

            using (var imageBuf = GpuContext.Instance.Accelerator.Allocate<uint>(slice.Extent))
            {
                PiсConvertFromByteKernel(imageView.GetSliceView(0).AsLinearView().Extent, imageBuf.AsLinearView(), slice.AsLinearView(), 1.0, 0);
                GpuContext.Instance.Accelerator.Synchronize();

                buff = imageBuf.GetAsArray();
            }

            return (outputBuf, buff);
        }

        private static readonly Action<Index, ArrayView<byte>, byte, byte> ThresholdingKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<byte>, byte, byte>(Kernels.Thresholding);

        private static readonly Action<Index, ArrayView<float>, ArrayView<byte>> FloatToByteKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<float>, ArrayView<byte>>(Kernels.FloatToByte);

        private static readonly Action<Index2, ArrayView3D<float>, ArrayView3D<byte>> PearsonKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView3D<float>, ArrayView3D<byte>>(CorrelationKernels.CorrelationMap);

        private static readonly Action<Index, ArrayView<float>, float> NormalizeKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<float>, float>(Kernels.NormalizeValues);

        public static CorrelationData CalculatePearsonCorrelation(ArrayView3D<byte> imageView, byte lowThreshold = 0, byte highThreshold = 0)
        {
            var result = new CorrelationData();

            var slice = imageView.GetSliceView(0);
            var comboSlice = new Index3(slice.Width, slice.Height, 3);

            using (var calcBuf = GpuContext.Instance.Accelerator.Allocate<float>(comboSlice))
            using (var byteBuf = GpuContext.Instance.Accelerator.Allocate<byte>(comboSlice))
            using (var imageBuf = GpuContext.Instance.Accelerator.Allocate<uint>(comboSlice))
            {
                PearsonKernel(slice.Extent, calcBuf.View, imageView);
                GpuContext.Instance.Accelerator.Synchronize();

                var maxCorrelation = calcBuf.GetAsArray().Max();
                NormalizeKernel(calcBuf.Extent.Size, calcBuf.AsLinearView(), maxCorrelation);
                GpuContext.Instance.Accelerator.Synchronize();

                var maxGradientValue = calcBuf.GetAsArray().Skip(slice.Extent.Size).Take(slice.Extent.Size).Max();
                NormalizeKernel(slice.Extent.Size, calcBuf.GetSliceView(1).AsLinearView(), maxGradientValue);
                GpuContext.Instance.Accelerator.Synchronize();

                FloatToByteKernel(calcBuf.Extent.Size, calcBuf.AsLinearView(), byteBuf.AsLinearView());
                GpuContext.Instance.Accelerator.Synchronize();

                if (highThreshold != 0 || lowThreshold != 0)
                {
                    ThresholdingKernel(byteBuf.Extent.Size, byteBuf.AsLinearView(), lowThreshold, highThreshold);
                    GpuContext.Instance.Accelerator.Synchronize();
                }

                result.rawPicture = byteBuf.GetAsArray().Skip(slice.Extent.Size * 2).ToArray();

                PiсConvertFromByteKernel(slice.Extent.Size, imageBuf.GetSliceView(0).AsLinearView(), byteBuf.GetSliceView(0).AsLinearView(), 1.0, 0);
                PiсConvertFromByteKernel(slice.Extent.Size, imageBuf.GetSliceView(1).AsLinearView(), byteBuf.GetSliceView(1).AsLinearView(), 1.0, 0);
                PiсConvertFromByteKernel(slice.Extent.Size, imageBuf.GetSliceView(2).AsLinearView(), byteBuf.GetSliceView(2).AsLinearView(), 1.0, 0);
                GpuContext.Instance.Accelerator.Synchronize();

                var img = imageBuf.GetAsArray();
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

        private static readonly Action<Index2, ArrayView2D<uint>, ArrayView3D<short>, int, int, int, short> CalPseudoColorKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<uint>, ArrayView3D<short>, int, int, int, short>(Kernels.CalculatePseudoColor);

        public static uint[] CalculatePseudoColor(ArrayView3D<short> imageView, int redBand, int greenBand, int blueBand, short max)
        {
            uint[] result;
            var sliceIndex = imageView.GetSliceView(0).Extent;
            using (var imageBuf = GpuContext.Instance.Accelerator.Allocate<uint>(sliceIndex))
            {
                CalPseudoColorKernel(sliceIndex, imageBuf.View, imageView, redBand, greenBand, blueBand, max);
                GpuContext.Instance.Accelerator.Synchronize();

                result = imageBuf.GetAsArray();
            }

            return result;
        }

        private static readonly Action<Index2, ArrayView2D<uint>, ArrayView3D<short>, int> CalcSliceBandKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<uint>, ArrayView3D<short>, int>(Kernels.CalculateSliceBand);

        private static readonly Action<Index, ArrayView2D<uint>, ArrayView3D<short>, int, int> CalcSlicesKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView2D<uint>, ArrayView3D<short>, int, int>(Kernels.CalculateSlices);

        public static uint[] CalcScanlineImage(ArrayView3D<short> imageView, int band, int row, int column)
        {
            uint[] data;

            Index2 index = new Index2(imageView.Extent.X + imageView.Extent.Z, imageView.Extent.Y + imageView.Extent.Z);
            var slice = imageView.GetSliceView(0).Extent;

            using (var imageBuf = GpuContext.Instance.Accelerator.Allocate<uint>(index))
            {
                CalcSliceBandKernel(slice, imageBuf, imageView, band);
                GpuContext.Instance.Accelerator.Synchronize();

                CalcSlicesKernel(imageView.Extent.Z, imageBuf, imageView, row, column);
                GpuContext.Instance.Accelerator.Synchronize();

                data = imageBuf.GetAsArray();
            }

            return data;
        }
    }

    public static class EdgeDetectionWrapper
    {
        private static readonly Action<Index2, ArrayView3D<float>, ArrayView3D<short>, short> CalculateSignatureLengthDerivativeKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView3D<float>, ArrayView3D<short>, short>(EdgeDetectionKernels.CalculateSignatureLengthDerivative);

        private static readonly Action<Index2, ArrayView3D<float>, ArrayView3D<short>, short> CalculateSignatureLengthDerivativeWithNormalizationKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView3D<float>, ArrayView3D<short>, short>(EdgeDetectionKernels.CalculateSignatureLengthDerivativeWithNormalization);

        private static readonly Action<Index2, ArrayView3D<float>, ArrayView3D<byte>> BCalculateSignatureLengthDerivativeKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView3D<float>, ArrayView3D<byte>>(EdgeDetectionKernels.CalculateSignatureLengthDerivative);

        private static readonly Action<Index2, ArrayView3D<float>, ArrayView3D<byte>> BCalculateSignatureLengthDerivativeWithNormalizationKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView3D<float>, ArrayView3D<byte>>(EdgeDetectionKernels.CalculateSignatureLengthDerivativeWithNormalization);

        private static readonly Action<Index2, ArrayView3D<float>, int, float> NormalizeLengthMapKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView3D<float>, int, float>(EdgeDetectionKernels.NormalizeLengthMap);

        private static readonly Action<Index, ArrayView<float>, ArrayView<byte>> FloatToByteKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<float>, ArrayView<byte>>(Kernels.FloatToByte);

        private static readonly Action<Index2, ArrayView2D<uint>, ArrayView2D<byte>, ArrayView2D<byte>, ArrayView2D<byte>, ArrayView2D<byte>> _accumulateEdgesKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index2, ArrayView2D<uint>, ArrayView2D<byte>, ArrayView2D<byte>, ArrayView2D<byte>, ArrayView2D<byte>>(EdgeDetectionKernels.AccumulateEdges);

        private static readonly Action<Index, ArrayView<uint>, ArrayView<byte>, double, short> PinConvertFromByteKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<uint>, ArrayView<byte>, double, short>(Kernels.PicConvertionByte);

        private static readonly Action<Index, ArrayView<byte>, byte, byte> ThresholdingKernel =
            GpuContext.Instance.Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<byte>, byte, byte>(Kernels.Thresholding);

        public static PicturesData CalculateSignatureLengthDerivative(ArrayView3D<short> imageView, bool normalize, short maxValue, byte threshold = 0)
        {
            PicturesData result = new PicturesData();
            var slice = imageView.GetSliceView(0);
            var picIndex = new Index3(slice.Extent.X, slice.Extent.Y, 3);

            using (var calcBuffer = GpuContext.Instance.Accelerator.Allocate<float>(picIndex))
            using (var byteBuf = GpuContext.Instance.Accelerator.Allocate<byte>(picIndex))
            using (var imageBuffer = GpuContext.Instance.Accelerator.Allocate<uint>(picIndex))
            {
                if (normalize)
                    CalculateSignatureLengthDerivativeWithNormalizationKernel(slice.Extent, calcBuffer.View, imageView, maxValue);
                else
                    CalculateSignatureLengthDerivativeKernel(slice.Extent, calcBuffer.View, imageView, maxValue);
                GpuContext.Instance.Accelerator.Synchronize();

                var array = calcBuffer.GetAsArray();
                var maxBand1 = array.Take(slice.Extent.Size).Max();
                var maxBand2 = array.Skip(slice.Extent.Size).Take(slice.Extent.Size).Max();
                var maxBand3 = XMath.Max(maxBand1, maxBand2);

                NormalizeLengthMapKernel(slice.Extent, calcBuffer, 0, maxBand1); GpuContext.Instance.Accelerator.Synchronize();
                NormalizeLengthMapKernel(slice.Extent, calcBuffer, 1, maxBand2); GpuContext.Instance.Accelerator.Synchronize();
                NormalizeLengthMapKernel(slice.Extent, calcBuffer, 2, maxBand3); GpuContext.Instance.Accelerator.Synchronize();

                FloatToByteKernel(calcBuffer.Extent.Size, calcBuffer.AsLinearView(), byteBuf.AsLinearView());
                GpuContext.Instance.Accelerator.Synchronize();

                result.rawPicture = byteBuf.GetAsArray().Skip(slice.Extent.Size * 2).ToArray();

                if (threshold != 0)
                {
                    ThresholdingKernel(slice.Extent.Size, byteBuf.GetSliceView(2).AsLinearView(), 0, threshold);
                }

                PinConvertFromByteKernel(slice.Extent.Size, imageBuffer.GetSliceView(0).AsLinearView(), byteBuf.GetSliceView(0).AsLinearView(), 1.0, 0);
                PinConvertFromByteKernel(slice.Extent.Size, imageBuffer.GetSliceView(1).AsLinearView(), byteBuf.GetSliceView(1).AsLinearView(), 1.0, 0);
                PinConvertFromByteKernel(slice.Extent.Size, imageBuffer.GetSliceView(2).AsLinearView(), byteBuf.GetSliceView(2).AsLinearView(), 1.0, 0);
                GpuContext.Instance.Accelerator.Synchronize();

                var img = imageBuffer.GetAsArray();
                var size = slice.Extent.Size;

                var buf = new uint[size];
                Array.Copy(img, 0, buf, 0, size);
                result.xPicture = buf;

                buf = new uint[size];
                Array.Copy(img, size, buf, 0, size);
                result.yPicture = buf;

                buf = new uint[size];
                Array.Copy(img, size * 2, buf, 0, size);
                result.xyPicture = buf;
            }

            return result;
        }

        public static PicturesData CalculateSignatureLengthDerivative(ArrayView3D<byte> imageView, bool normalize)
        {
            PicturesData result = new PicturesData();
            var slice = imageView.GetSliceView(0);
            var picIndex = new Index3(slice.Extent.X, slice.Extent.Y, 3);

            using (var calcBuffer = GpuContext.Instance.Accelerator.Allocate<float>(picIndex))
            using (var byteBuf = GpuContext.Instance.Accelerator.Allocate<byte>(picIndex))
            using (var imageBuffer = GpuContext.Instance.Accelerator.Allocate<uint>(picIndex))
            {
                if (normalize)
                    BCalculateSignatureLengthDerivativeWithNormalizationKernel(slice.Extent, calcBuffer.View, imageView);
                else
                    BCalculateSignatureLengthDerivativeKernel(slice.Extent, calcBuffer.View, imageView);
                GpuContext.Instance.Accelerator.Synchronize();

                var array = calcBuffer.GetAsArray();
                var maxBand1 = array.Take(slice.Extent.Size).Max();
                var maxBand2 = array.Skip(slice.Extent.Size).Take(slice.Extent.Size).Max();
                //var maxBand3 = array.Skip(slice.Extent.Size * 2).Max();
                var maxBand3 = XMath.Max(maxBand1, maxBand2);

                NormalizeLengthMapKernel(slice.Extent, calcBuffer, 0, maxBand1); GpuContext.Instance.Accelerator.Synchronize();
                NormalizeLengthMapKernel(slice.Extent, calcBuffer, 1, maxBand2); GpuContext.Instance.Accelerator.Synchronize();
                NormalizeLengthMapKernel(slice.Extent, calcBuffer, 2, maxBand3); GpuContext.Instance.Accelerator.Synchronize();

                FloatToByteKernel(calcBuffer.Extent.Size, calcBuffer.AsLinearView(), byteBuf.AsLinearView());
                GpuContext.Instance.Accelerator.Synchronize();

                result.rawPicture = byteBuf.GetAsArray().Skip(slice.Extent.Size * 2).ToArray();


                PinConvertFromByteKernel(slice.Extent.Size, imageBuffer.GetSliceView(0).AsLinearView(), byteBuf.GetSliceView(0).AsLinearView(), 1.0, 0);
                PinConvertFromByteKernel(slice.Extent.Size, imageBuffer.GetSliceView(1).AsLinearView(), byteBuf.GetSliceView(1).AsLinearView(), 1.0, 0);
                PinConvertFromByteKernel(slice.Extent.Size, imageBuffer.GetSliceView(2).AsLinearView(), byteBuf.GetSliceView(2).AsLinearView(), 1.0, 0);
                GpuContext.Instance.Accelerator.Synchronize();

                var img = imageBuffer.GetAsArray();
                var size = slice.Extent.Size;

                var buf = new uint[size];
                Array.Copy(img, 0, buf, 0, size);
                result.xPicture = buf;

                buf = new uint[size];
                Array.Copy(img, size, buf, 0, size);
                result.yPicture = buf;

                buf = new uint[size];
                Array.Copy(img, size * 2, buf, 0, size);
                result.xyPicture = buf;
            }

            return result;
        }

        public static uint[] AccumulateEdges(ArrayView3D<short> imageShortView, ArrayView3D<byte> imageByteView, byte[] cannyData, byte pearsonThreshold, short maxValue)
        {
            uint[] result;

            var pearsonData = CalculationWrappers.CalculatePearsonCorrelation(imageByteView, 0, pearsonThreshold).rawPicture;
            var signLengthByte = CalculateSignatureLengthDerivative(imageByteView, true).rawPicture;
            var signLengthShort = CalculateSignatureLengthDerivative(imageShortView, true, maxValue).rawPicture;

            var picIndex = imageByteView.GetSliceView(0).Extent;

            using (var pearsBuffer = GpuContext.Instance.Accelerator.Allocate<byte>(picIndex))
            using (var signLengthByteBuffer = GpuContext.Instance.Accelerator.Allocate<byte>(picIndex))
            using (var signLengthShortBuffer = GpuContext.Instance.Accelerator.Allocate<byte>(picIndex))
            using (var cannyBuffer = GpuContext.Instance.Accelerator.Allocate<byte>(picIndex))
            using (var imgBuffer = GpuContext.Instance.Accelerator.Allocate<uint>(picIndex))
            {
                pearsBuffer.CopyFrom(pearsonData, 0, Index2.Zero, picIndex.Size);
                signLengthByteBuffer.CopyFrom(signLengthByte, 0, Index2.Zero, picIndex.Size);
                signLengthShortBuffer.CopyFrom(signLengthShort, 0, Index2.Zero, picIndex.Size);
                cannyBuffer.CopyFrom(cannyData, 0, Index2.Zero, picIndex.Size);

                _accumulateEdgesKernel(picIndex, imgBuffer.View, pearsBuffer.View, cannyBuffer.View, signLengthByteBuffer.View, signLengthShortBuffer.View);
                GpuContext.Instance.Accelerator.Synchronize();

                result = imgBuffer.GetAsArray();
            }

            return result;
        }
    }
}
