using ILGPU;
using System;
using System.Collections.Generic;
using System.Text;

namespace AvaloniaMVVM.Kernels
{
    public class HistogramKernels
    {
        public static void StandardDeviation(Index index, ArrayView<double> result, ArrayView3D<short> input)
        {
            long sum = 0;

            var band = input.GetSliceView(index).AsLinearView();
            for (int i = 0; i < band.Length; i++)
            {
                sum += band[i];
            }

            double mean = sum / band.Length;
            double dividend = 0;
            for (int i = 0; i < band.Length; i++)
            {
                dividend += XMath.Pow(band[i] - mean, 2);
            }

            result[index] = XMath.Sqrt(dividend / band.Length);
        }

        public static void CalculateDeviationImage(Index index, ArrayView2D<short> result, ArrayView<double> input, double imageWidth, double imageHeight, int maxDeviation)
        {
            int x = (int)(imageWidth * (index.X / (double)input.Length));
            int y = (int)(imageHeight * (input[index] / (double)maxDeviation));
            for (int i = 0; i < result.Height; i++)
            {
                result[x, result.Height - i - 1] = i > y ? short.MaxValue : (short)0;
            }
        }

        public static void CalculateHistogram(Index3 index, ArrayView<int> result, ArrayView3D<short> input)
        {
            var val = input[index];
            if (val < 0)
                val = 0;
            Atomic.Add(ref result[(int)val], 1);
        }

        public static void CalculateHistogramImage(Index index, ArrayView2D<short> result, ArrayView<int> input, double imageWidth, double imageHeight, int max)
        {
            if (index.X == 0)   // because zeroes is too huge, and not interesting for histogram image
                return;

            imageHeight--;
            int x = (int)(imageWidth * (index.X / (double)input.Length));
            int y = (int)(imageHeight * (input[index] / (double)max));
            for (int i = 0; i < result.Height; i++)
            {
                result[x, result.Height - i - 1] = i > y ? short.MaxValue : (short)0;
            }
        }
    }
}
