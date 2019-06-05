using ILGPU;
using System;
using System.Collections.Generic;
using System.Text;

namespace AvaloniaMVVM.Kernels
{
    public class BrightnessKernels
    {
        public static void CalculateBrightnessStats(Index index, ArrayView3D<short> input, ArrayView<double> MeanBrightness, ArrayView<short> MaxBrightness, ArrayView<double> StandartDeviation)
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
            MeanBrightness[index] = mean;
            MaxBrightness[index] = max;

            double dividend = 0;
            for (int i = 0; i < band.Length; i++)
            {
                var val = band[i];
                if (val < 0)
                    val = 0;

                dividend += XMath.Pow(XMath.Abs(val - mean), 2);
            }

            StandartDeviation[index] = XMath.Sqrt(dividend / band.Length);
        }

        public static void CalculateBrightnessMap(Index index, ArrayView<double> MeanBrightness, ArrayView<short> MaxBrightness, ArrayView<double> StandartDeviation, ArrayView2D<int> Image, int meanBrColor, int maxBrColor, int devBrColor, double heightScale)
        {
            int xCoord1 = (int)((index / (float)MaxBrightness.Extent.Size) * Image.Width);
            int xCoord2 = (int)(((index + 1) / (float)MaxBrightness.Extent.Size) * Image.Width - 1);

            for (int i = xCoord1; i < xCoord2; i++)
            {
                Image[i, (int)(Image.Height - MeanBrightness[index] * heightScale)] = meanBrColor;
                Image[i, (int)(Image.Height - MaxBrightness[index] * heightScale)] = maxBrColor;
                Image[i, (int)(Image.Height - StandartDeviation[index] * heightScale)] = devBrColor;
            }
        }
    }
}
