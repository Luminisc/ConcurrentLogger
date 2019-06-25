using ILGPU;

namespace AvaloniaMVVM.Kernels
{
    public class Kernels
    {
        public static void PicConvertion(Index index, ArrayView<uint> buf1, ArrayView<short> buf2, double mult = 1, short min = 0)
        {
            byte rad = (byte)((buf2[index] - min) * mult);
            buf1[index] = (uint)(rad + (rad << 8) + (rad << 16) + (255 << 24));
        }

        public static void PicConvertionByte(Index index, ArrayView<uint> buf1, ArrayView<byte> buf2, double mult = 1, short min = 0)
        {
            byte rad = buf2[index];

            if (rad > 254)
                rad = 254; // because there are issue with saving of white pixels

            buf1[index] = (uint)(rad + (rad << 8) + (rad << 16) + (255 << 24));
        }

        #region Correlation
        // Correct one!
        public static double PearsonCorrelation(ArrayView3D<byte> bufIn, Index2 pixel1, Index2 pixel2)
        {
            var minLength = bufIn.Depth;
            int sum1 = 0, sum2 = 0;
            float mean1 = 0, mean2 = 0;

            for (int i = 0; i < minLength; i++)
            {
                byte x = bufIn[pixel1.X, pixel1.Y, i];
                byte y = bufIn[pixel2.X, pixel2.Y, i];
                sum1 += x;
                sum2 += y;
            }
            mean1 = sum1 / minLength;
            mean2 = sum2 / minLength;

            float covariation = 0;
            float xDerSqrSum = 0;
            float yDerSqrSum = 0;

            for (int i = 0; i < minLength; i++)
            {
                byte x = bufIn[pixel1.X, pixel1.Y, i];
                byte y = bufIn[pixel2.X, pixel2.Y, i];

                float xDer = x - mean1;
                float yDer = y - mean2;

                covariation += xDer * yDer;
                xDerSqrSum += xDer * xDer;
                yDerSqrSum += yDer * yDer;

            }

            return covariation / XMath.Sqrt(xDerSqrSum * yDerSqrSum);
        }

        public static double PearsonCorrelation(ArrayView3D<short> bufIn, Index2 pixel1, Index2 pixel2)
        {
            var minLength = bufIn.Depth;
            int sum1 = 0, sum2 = 0;
            float mean1 = 0, mean2 = 0;

            for (int i = 0; i < minLength; i++)
            {
                short x = bufIn[pixel1.X, pixel1.Y, i];
                short y = bufIn[pixel2.X, pixel2.Y, i];
                sum1 += x;
                sum2 += y;
            }
            mean1 = sum1 / minLength;
            mean2 = sum2 / minLength;

            float covariation = 0;
            float xDerSqrSum = 0;
            float yDerSqrSum = 0;

            for (int i = 0; i < minLength; i++)
            {
                short x = bufIn[pixel1.X, pixel1.Y, i];
                short y = bufIn[pixel2.X, pixel2.Y, i];

                float xDer = x - mean1;
                float yDer = y - mean2;

                covariation += xDer * yDer;
                xDerSqrSum += xDer * xDer;
                yDerSqrSum += yDer * yDer;

            }

            return covariation / XMath.Sqrt(xDerSqrSum * yDerSqrSum);
        }

        public static double PearsonSobelCorrelation(ArrayView3D<byte> bufIn, Index2 pixel1, Index2 pixel2)
        {
            var minLength = bufIn.Depth;
            int sum1 = 0, sum2 = 0;
            float mean1 = 0, mean2 = 0;

            for (int i = 0; i < minLength; i++)
            {
                short x = bufIn[pixel1.X, pixel1.Y, i];
                short y = bufIn[pixel2.X, pixel2.Y, i];
                sum1 += x;
                sum2 += y;
            }
            mean1 = sum1 / minLength;
            mean2 = sum2 / minLength;

            float covariation = 0;
            float xDerSqrSum = 0;
            float yDerSqrSum = 0;

            for (int i = 0; i < minLength; i++)
            {
                short x = bufIn[pixel1.X, pixel1.Y, i];
                short y = bufIn[pixel2.X, pixel2.Y, i];

                float xDer = x - mean1;
                float yDer = y - mean2;

                covariation += xDer * yDer;
                xDerSqrSum += xDer * xDer;
                yDerSqrSum += yDer * yDer;

            }

            return covariation / XMath.Sqrt(xDerSqrSum * yDerSqrSum);
        }

        public static void HorizontalCorrelationMap(Index2 index, ArrayView2D<double> result, ArrayView3D<short> bufIn1)
        {
            if (index.X + 1 == bufIn1.Width)
            {
                result[index] = 1.0;
                return;
            }

            var index1 = new Index2(index.X, index.Y);
            var index2 = new Index2(index.X + 1, index.Y);

            result[index] = PearsonCorrelation(bufIn1, index1, index2);
        }

        public static void VerticalCorrelationMap(Index2 index, ArrayView2D<double> result, ArrayView3D<short> bufIn1)
        {
            if (index.Y + 1 == bufIn1.Height)
            {
                result[index] = 1.0;
                return;
            }

            var index1 = new Index2(index.X, index.Y);
            var index2 = new Index2(index.X, index.Y + 1);

            result[index] = PearsonCorrelation(bufIn1, index1, index2);
        }

        public static void DiagLeftBottomCorrelationMap(Index2 index, ArrayView2D<double> result, ArrayView3D<short> bufIn1)
        {
            if (index.X + 1 == bufIn1.Width || index.Y + 1 == bufIn1.Height)
            {
                result[index] = 1.0;
                return;
            }

            var index1 = new Index2(index.X, index.Y);
            var index2 = new Index2(index.X + 1, index.Y + 1);

            result[index] = PearsonCorrelation(bufIn1, index1, index2);
        }

        public static void DiagRightBottomCorrelationMap(Index2 index, ArrayView2D<double> result, ArrayView3D<short> bufIn1)
        {
            if (index.X + 1 == bufIn1.Width || index.Y + 1 == bufIn1.Height)
            {
                result[index] = 1.0;
                return;
            }

            var index1 = new Index2(index.X + 1, index.Y);
            var index2 = new Index2(index.X, index.Y + 1);

            result[index] = PearsonCorrelation(bufIn1, index1, index2);
        }

        public static void BuildCorrelationMap(Index2 index, ArrayView2D<double> result,
            ArrayView2D<double> bufIn1, ArrayView2D<double> bufIn2,
            ArrayView2D<double> bufIn3, ArrayView2D<double> bufIn4)
        {
            result[index] = XMath.Min(XMath.Min(XMath.Min(bufIn1[index], bufIn2[index]), bufIn3[index]), bufIn4[index]);
        }

        public static void CorrelationMapToRgba32(Index2 index, ArrayView2D<uint> result, ArrayView2D<double> bufIn1)
        {
            double r = bufIn1[index];
            byte rad = 0;

            r = XMath.Abs(r);
            if (r > 1) r = 1;

            rad = (byte)(255 * r);
            if (rad == 255)
                rad = 254; // because there are issue with saving of white pixels

            result[index] = (uint)(rad + (rad << 8) + (rad << 16) + (255 << 24));
        }
        #endregion


        //static readonly float[,] sobelXKern = new float[3, 3]
        //{
        //    { -1,0,1 },
        //    { -2,0,2 },
        //    { -1,0,1 }
        //};

        //static readonly float[,] sobelYKern = new float[3, 3]
        //{
        //    { -1,-2,-1 },
        //    { 0, 0, 0 },
        //    { 1, 2, 1 }
        //};

        #region Sobel
        public static void CalculateHorizontalSobel(Index2 index, ArrayView3D<short> result, ArrayView3D<short> image)
        {
            if (index.X == 0 || index.Y == 0
                || index.X == image.Width - 1 || index.Y == image.Height - 1)
            {
                for (int i = 0; i < image.Depth; i++)
                {
                    result[index.X, index.Y, i] = 32000;
                }
                return;
            }

            for (int i = 0; i < image.Depth; i++)
            {
                var kern = image.GetSliceView(i).GetSubView(new Index2(index.X - 1, index.Y - 1), new Index2(3, 3));
                short val = 0;

                val += (short)(kern[0, 0] * -1);
                val += (short)(kern[2, 0] * 1);
                val += (short)(kern[0, 1] * -2);
                val += (short)(kern[2, 1] * 2);
                val += (short)(kern[0, 2] * -1);
                val += (short)(kern[2, 2] * 1);

                result[index.X, index.Y, i] = val;
            }
        }

        public static void CalculateVerticalSobel(Index2 index, ArrayView3D<short> result, ArrayView3D<short> image)
        {
            if (index.X == 0 || index.Y == 0
                || index.X == image.Width - 1 || index.Y == image.Height - 1)
            {
                for (int i = 0; i < image.Depth; i++)
                {
                    result[index.X, index.Y, i] = 30000;
                }
                return;
            }

            //var sobelYKern = new short[3, 3]
            //{
            //    { -1,-2,-1 },
            //    { 0, 0, 0 },
            //    { 1, 2, 1 }
            //};

            for (int i = 0; i < image.Depth; i++)
            {
                var kern = image.GetSubView(new Index3(index.X - 1, index.Y - 1, i), new Index3(3, 3, 1)).GetSliceView(0);
                short val = 0;

                val += (short)(kern[0, 0] * -1);
                val += (short)(kern[1, 0] * -2);
                val += (short)(kern[2, 0] * -1);
                val += (short)(kern[0, 2] * 1);
                val += (short)(kern[1, 2] * 2);
                val += (short)(kern[2, 2] * 1);
                result[index.X, index.Y, i] = val;
            }
        }

        public static void AccumulateSobelMap(Index3 index, ArrayView3D<short> result, ArrayView3D<short> hSobel, ArrayView3D<short> vSobel)
        {
            result[index] = KernelHelpers.Magnitude(hSobel[index], vSobel[index]);
        }

        public static void CalculateSobel(Index2 index, ArrayView2D<short> result, ArrayView3D<short> accumulatedSobel)
        {
            var val = KernelHelpers.Magnitude(accumulatedSobel, index);
            result[index] = val;
        }
        #endregion

        public static void CalculateSliceBand(Index2 index, ArrayView2D<uint> result, ArrayView3D<short> input, int band)
        {
            var ind = new Index2(index.X, index.Y + input.Depth);
            uint val = (uint)(input[index.X, index.Y, band] / 9000f * 255);
            result[ind] = (uint)((val) + (val << 8) + (val << 16) + (255 << 24));
        }

        public static void CalculateSlices(Index index, ArrayView2D<uint> result, ArrayView3D<short> input, int row, int column)
        {
            for (int i = 0; i < input.Width; i++)
            {
                uint val = (uint)(input[i, row, index.X] / 9000f * 255);
                result[i + 1 /*+ index.X*/, input.Depth /*- index.X*/] = (uint)((val) + (val << 8) + (val << 16) + (255 << 24));
            }
            for (int i = 0; i < input.Height; i++)
            {
                uint val = (uint)(input[column, i, index.X] / 9000f * 255);
                result[input.Width /*+ index.X*/, input.Depth - 1 /*- index.X*/ + i] = (uint)((val) + (val << 8) + (val << 16) + (255 << 24));
            }
        }

        #region Utils
        public static void NoOp(Index index, ArrayView<short> result, ArrayView<short> input)
        {
            result[index] = (short)(input[index] * 1);
        }

        public static void Memset(Index index, ArrayView<int> input, int value)
        {
            input[index] = value;
        }

        public static void ConvertToByte(Index3 index, ArrayView3D<byte> output, ArrayView3D<short> input, short maxValue)
        {
            var val = input[index];
            val = XMath.Clamp(val, (short)0, maxValue);
            byte result = (byte)((val / (float)maxValue) * 255);
            output[index] = result;
        }

        public static void NormalizeValues(Index index, ArrayView<float> input, float maxValue)
        {
            input[index] /= maxValue;
        }

        public static void FloatToByte(Index index, ArrayView<float> input, ArrayView<byte> output)
        {
            var val = input[index];
            if (val > 1.0f)
                val = 1.0f;
            if (val < 0)
                val = 0;
            output[index] = (byte)(val * 255);
        }

        public static void Thresholding(Index index, ArrayView<byte> input, /*remove*/byte lowThreshold, byte highThreshold)
        {
            // TODO: correct two-side thresholding
            var val = input[index];

            if (val > highThreshold)
                val = 255;
            else
                val = 0;

            input[index] = val;
        }

        public static void CalculatePseudoColor(Index2 index, ArrayView2D<uint> output, ArrayView3D<short> input, int redBand, int greenBand, int blueBand, short max)
        {
            uint r = (uint)(XMath.Clamp(input[index.X, index.Y, redBand], (short)0, max) / (float)max * 255);
            uint g = (uint)(XMath.Clamp(input[index.X, index.Y, greenBand], (short)0, max) / (float)max * 255);
            uint b = (uint)(XMath.Clamp(input[index.X, index.Y, blueBand], (short)0, max) / (float)max * 255);

            output[index] = r + (g << 8) + (b << 16) + ((uint)255 << 24);
        }
        #endregion
    }
}
