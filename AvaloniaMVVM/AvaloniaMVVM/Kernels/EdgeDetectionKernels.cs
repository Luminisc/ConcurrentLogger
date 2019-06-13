using ILGPU;
using System;
using System.Collections.Generic;
using System.Text;

namespace AvaloniaMVVM.Kernels
{
    public class EdgeDetectionKernels
    {
        public static void CalculateSignatureLengthDerivativeWithNormalization(Index2 index, ArrayView3D<float> output, ArrayView3D<short> input, short maxValue)
        {
            var x = index.X;
            var y = index.Y;
            if (x == 0)
                x = 1;
            if (x == input.Width - 1)
                x = input.Width - 2;
            if (y == 0)
                y = 1;
            if (y == input.Height - 1)
                y = input.Height - 2;

            var depth = input.Depth;

            int x1ComponentSum = 0;
            int x2ComponentSum = 0;

            int y1ComponentSum = 0;
            int y2ComponentSum = 0;

            for (int i = 0; i < depth; i++)
            {
                var val1 = XMath.Clamp(input[x - 1, y, i], (short)0, maxValue);
                var val2 = XMath.Clamp(input[x + 1, y, i], (short)0, maxValue);
                x1ComponentSum += val1 * val1;
                x2ComponentSum += val2 * val2;

                val1 = XMath.Clamp(input[x, y - 1, i], (short)0, maxValue);
                val2 = XMath.Clamp(input[x, y + 1, i], (short)0, maxValue);
                y1ComponentSum += val1 * val1;
                y2ComponentSum += val2 * val2;
            }

            float x1Length = XMath.Sqrt(x1ComponentSum);
            float x2Length = XMath.Sqrt(x2ComponentSum);
            float y1Length = XMath.Sqrt(y1ComponentSum);
            float y2Length = XMath.Sqrt(y2ComponentSum);

            float xDiffComponentSum = 0;
            float yDiffComponentSum = 0;

            for (int i = 0; i < depth; i++)
            {
                var val1 = XMath.Clamp(input[x - 1, y, i], (short)0, maxValue);
                var val2 = XMath.Clamp(input[x + 1, y, i], (short)0, maxValue);
                float xDiff = (val2 / x2Length) - (val1 / x1Length);
                xDiffComponentSum += xDiff * xDiff;

                val1 = XMath.Clamp(input[x, y - 1, i], (short)0, maxValue);
                val2 = XMath.Clamp(input[x, y + 1, i], (short)0, maxValue);
                float yDiff = (val2 / y2Length) - (val1 / y1Length);
                yDiffComponentSum += yDiff * yDiff;
            }

            var xDiffLength = XMath.Sqrt(xDiffComponentSum);
            var yDiffLength = XMath.Sqrt(yDiffComponentSum);

            output[index.X, index.Y, 0] = xDiffLength;
            output[index.X, index.Y, 1] = yDiffLength;
            output[index.X, index.Y, 2] = XMath.Max(xDiffLength, yDiffLength);
        }

        public static void CalculateSignatureLengthDerivative(Index2 index, ArrayView3D<float> output, ArrayView3D<short> input, short maxValue)
        {
            var x = index.X;
            var y = index.Y;
            if (x == 0)
                x = 1;
            if (x == input.Width - 1)
                x = input.Width - 2;
            if (y == 0)
                y = 1;
            if (y == input.Height - 1)
                y = input.Height - 2;

            var depth = input.Depth;

            float xDiffComponentSum = 0;
            float yDiffComponentSum = 0;

            for (int i = 0; i < depth; i++)
            {
                var val1 = XMath.Clamp(input[x - 1, y, i], (short)0, maxValue);
                var val2 = XMath.Clamp(input[x + 1, y, i], (short)0, maxValue);
                var xDiff = val2 - val1;
                xDiffComponentSum += xDiff * xDiff;

                val1 = XMath.Clamp(input[x, y - 1, i], (short)0, maxValue);
                val2 = XMath.Clamp(input[x, y + 1, i], (short)0, maxValue);
                var yDiff = val2 - val1;
                yDiffComponentSum += yDiff * yDiff;
            }

            var xDiffLength = XMath.Sqrt(xDiffComponentSum);
            var yDiffLength = XMath.Sqrt(yDiffComponentSum);

            output[index.X, index.Y, 0] = xDiffLength;
            output[index.X, index.Y, 1] = yDiffLength;
            output[index.X, index.Y, 2] = XMath.Max(xDiffLength, yDiffLength);
        }


        public static void CalculateSignatureLengthDerivativeWithNormalization(Index2 index, ArrayView3D<float> output, ArrayView3D<byte> input)
        {
            var x = index.X;
            var y = index.Y;
            if (x == 0)
                x = 1;
            if (x == input.Width - 1)
                x = input.Width - 2;
            if (y == 0)
                y = 1;
            if (y == input.Height - 1)
                y = input.Height - 2;

            var depth = input.Depth;

            int x1ComponentSum = 0;
            int x2ComponentSum = 0;

            int y1ComponentSum = 0;
            int y2ComponentSum = 0;

            for (int i = 0; i < depth; i++)
            {
                var val1 = input[x - 1, y, i]; if (val1 < 0) val1 = 0;
                var val2 = input[x + 1, y, i]; if (val2 < 0) val1 = 0;
                x1ComponentSum += val1 * val1;
                x2ComponentSum += val2 * val2;

                val1 = input[x, y - 1, i]; if (val1 < 0) val1 = 0;
                val2 = input[x, y + 1, i]; if (val2 < 0) val1 = 0;
                y1ComponentSum += val1 * val1;
                y2ComponentSum += val2 * val2;
            }

            float x1Length = XMath.Sqrt(x1ComponentSum);
            float x2Length = XMath.Sqrt(x2ComponentSum);
            float y1Length = XMath.Sqrt(y1ComponentSum);
            float y2Length = XMath.Sqrt(y2ComponentSum);

            float xDiffComponentSum = 0;
            float yDiffComponentSum = 0;

            for (int i = 0; i < depth; i++)
            {
                var val1 = input[x - 1, y, i]; if (val1 < 0) val1 = 0;
                var val2 = input[x + 1, y, i]; if (val2 < 0) val1 = 0;
                float xDiff = (val2 / x2Length) - (val1 / x1Length);
                xDiffComponentSum += xDiff * xDiff;

                val1 = input[x, y - 1, i]; if (val1 < 0) val1 = 0;
                val2 = input[x, y + 1, i]; if (val2 < 0) val1 = 0;
                float yDiff = (val2 / y2Length) - (val1 / y1Length);
                yDiffComponentSum += yDiff * yDiff;
            }

            var xDiffLength = XMath.Sqrt(xDiffComponentSum);
            var yDiffLength = XMath.Sqrt(yDiffComponentSum);

            output[index.X, index.Y, 0] = xDiffLength;
            output[index.X, index.Y, 1] = yDiffLength;
            output[index.X, index.Y, 2] = XMath.Max(xDiffLength, yDiffLength);
        }

        public static void CalculateSignatureLengthDerivative(Index2 index, ArrayView3D<float> output, ArrayView3D<byte> input)
        {
            var x = index.X;
            var y = index.Y;
            if (x == 0)
                x = 1;
            if (x == input.Width - 1)
                x = input.Width - 2;
            if (y == 0)
                y = 1;
            if (y == input.Height - 1)
                y = input.Height - 2;

            var depth = input.Depth;

            float xDiffComponentSum = 0;
            float yDiffComponentSum = 0;

            for (int i = 0; i < depth; i++)
            {
                var val1 = input[x - 1, y, i]; if (val1 < 0) val1 = 0;
                var val2 = input[x + 1, y, i]; if (val2 < 0) val1 = 0;
                var xDiff = val2 - val1;
                xDiffComponentSum += xDiff * xDiff;

                val1 = input[x, y - 1, i]; if (val1 < 0) val1 = 0;
                val2 = input[x, y + 1, i]; if (val2 < 0) val1 = 0;
                var yDiff = val2 - val1;
                yDiffComponentSum += yDiff * yDiff;
            }

            var xDiffLength = XMath.Sqrt(xDiffComponentSum);
            var yDiffLength = XMath.Sqrt(yDiffComponentSum);

            output[index.X, index.Y, 0] = xDiffLength;
            output[index.X, index.Y, 1] = yDiffLength;
            output[index.X, index.Y, 2] = XMath.Max(xDiffLength, yDiffLength);
        }


        public static void NormalizeLengthMap(Index2 index, ArrayView3D<float> output, int zIndex, float maxValue)
        {
            output[index.X, index.Y, zIndex] /= maxValue;
        }

        public static void AccumulateEdges(Index2 index, ArrayView2D<uint> output, ArrayView2D<byte> pearson, ArrayView2D<byte> canny, ArrayView2D<byte> signLengthByte, ArrayView2D<byte> signLengthShort)
        {
            uint r = (uint)XMath.Max(pearson[index] / 3, canny[index]); if (r > 254) r = 254;
            uint g = (uint)(XMath.Max(pearson[index] / 3, signLengthByte[index])); if (g > 254) g = 254;
            uint b = (uint)(XMath.Max(pearson[index] / 3, signLengthShort[index])); if (b > 254) b = 254;

            uint value = (uint)(r + (g << 8) + (b << 16) + (255 << 24));

            output[index] = value;
        }
    }


}
