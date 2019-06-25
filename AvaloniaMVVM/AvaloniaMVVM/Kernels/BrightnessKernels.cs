using ILGPU;

namespace AvaloniaMVVM.Kernels
{
    public class BrightnessKernels
    {
        public static void CalculateBrightnessStats(Index index, ArrayView3D<short> input, ArrayView<double> meanBrightness, ArrayView<short> maxBrightness, ArrayView<double> standartDeviation)
        {
            long sum = 0;

            short max = 0;

            var band = input.GetSliceView(index).AsLinearView();
            for (int i = 0; i < band.Length; i++)
            {
                var val = band[i];
                if (val < 0)
                    val = 0;
                //for deviation and mean sum
                sum += val;
                if (val > max)
                    max = val;
            }

            double mean = sum / (double)band.Length;
            meanBrightness[index] = mean;
            maxBrightness[index] = max;

            double dividend = 0;
            for (int i = 0; i < band.Length; i++)
            {
                var val = band[i];
                if (val < 0)
                    val = 0;

                dividend += XMath.Pow(XMath.Abs(val - mean), 2);
            }

            standartDeviation[index] = XMath.Sqrt(dividend / band.Length);
        }

        public static void CalculateBrightnessMap(Index index, ArrayView<double> meanBrightness, ArrayView<short> maxBrightness, ArrayView<double> standartDeviation, ArrayView2D<int> image, int meanBrColor, int maxBrColor, int devBrColor, double heightScale)
        {
            int xCoord1 = (int)((index / (float)maxBrightness.Extent.Size) * image.Width);
            int xCoord2 = (int)(((index + 1) / (float)maxBrightness.Extent.Size) * image.Width - 1);

            for (int i = xCoord1; i < xCoord2; i++)
            {
                image[i, (int)(image.Height - meanBrightness[index] * heightScale)] = meanBrColor;
                image[i, (int)(image.Height - maxBrightness[index] * heightScale)] = maxBrColor;
                image[i, (int)(image.Height - standartDeviation[index] * heightScale)] = devBrColor;
            }
        }
    }
}
