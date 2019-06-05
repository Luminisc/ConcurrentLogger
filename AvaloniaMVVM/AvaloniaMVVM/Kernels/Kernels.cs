﻿using ILGPU;
using System;
using System.Collections.Generic;
using System.Text;

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
            var val = buf2[index];
            byte rad = (byte)(255 - val);
            if (rad == 255)
                rad = 254; // because there are issue with saving of white pixels

            buf1[index] = (uint)(rad + (rad << 8) + (rad << 16) + (255 << 24));
        }

        #region Correlation
        // not correct!
        public static void PearsonCorrelation(Index index, VariableView<double> result, ArrayView<short> bufIn1, ArrayView<short> bufIn2)
        {
            double sx = 0.0;
            double sy = 0.0;
            double sxx = 0.0;
            double syy = 0.0;
            double sxy = 0.0;

            int n = 0;

            var minLength = Math.Min(bufIn1.Length, bufIn2.Length);
            for (int i = 0; i < minLength; i++)
            {
                short x = bufIn1[i];
                short y = bufIn2[i];

                n += 1;
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }

            double covariation = sxy / n - sx * sy / n / n;
            double sigmaX = Math.Sqrt(sxx / n - sx * sx / n / n);
            double sigmaY = Math.Sqrt(syy / n - sy * sy / n / n);

            result.Value = covariation / (sigmaX * sigmaY);
        }

        // Correct one!
        public static double PearsonCorrelation(ArrayView3D<byte> bufIn, Index2 pixel1, Index2 pixel2)
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

            var a = bufIn1.GetSubView(new Index3(index.X, index.Y, 0), new Index3(1, 1, bufIn1.Depth)).AsLinearView();
            var b = bufIn1.GetSubView(new Index3(index.X + 1, index.Y, 0), new Index3(1, 1, bufIn1.Depth)).AsLinearView();
            VariableView<double> res = result.GetVariableView(index);
            PearsonCorrelation(1, res, a, b);
            result[index] = res.Value;
        }

        public static void VerticalCorrelationMap(Index2 index, ArrayView2D<double> result, ArrayView3D<short> bufIn1)
        {
            if (index.Y + 1 == bufIn1.Height)
            {
                result[index] = 1.0;
                return;
            }

            var a = bufIn1.GetSubView(new Index3(index.X, index.Y, 0), new Index3(1, 1, bufIn1.Depth)).AsLinearView();
            var b = bufIn1.GetSubView(new Index3(index.X, index.Y + 1, 0), new Index3(1, 1, bufIn1.Depth)).AsLinearView();
            VariableView<double> res = result.GetVariableView(index);
            PearsonCorrelation(1, res, a, b);
            result[index] = res.Value;
        }

        public static void DiagLeftBottomCorrelationMap(Index2 index, ArrayView2D<double> result, ArrayView3D<short> bufIn1)
        {
            if (index.X + 1 == bufIn1.Width || index.Y + 1 == bufIn1.Height)
            {
                result[index] = 1.0;
                return;
            }

            var a = bufIn1.GetSubView(new Index3(index.X, index.Y, 0), new Index3(1, 1, bufIn1.Depth)).AsLinearView();
            var b = bufIn1.GetSubView(new Index3(index.X + 1, index.Y + 1, 0), new Index3(1, 1, bufIn1.Depth)).AsLinearView();
            VariableView<double> res = result.GetVariableView(index);
            PearsonCorrelation(1, res, a, b);
            result[index] = res.Value;
        }

        public static void DiagRightBottomCorrelationMap(Index2 index, ArrayView2D<double> result, ArrayView3D<short> bufIn1)
        {
            if (index.X + 1 == bufIn1.Width || index.Y + 1 == bufIn1.Height)
            {
                result[index] = 1.0;
                return;
            }

            var a = bufIn1.GetSubView(new Index3(index.X + 1, index.Y, 0), new Index3(1, 1, bufIn1.Depth)).AsLinearView();
            var b = bufIn1.GetSubView(new Index3(index.X, index.Y + 1, 0), new Index3(1, 1, bufIn1.Depth)).AsLinearView();
            VariableView<double> res = result.GetVariableView(index);
            PearsonCorrelation(1, res, a, b);
            result[index] = res.Value;
        }

        public static void BuildCorrelationMap(Index2 index, ArrayView2D<double> result,
            ArrayView2D<double> bufIn1, ArrayView2D<double> bufIn2,
            ArrayView2D<double> bufIn3, ArrayView2D<double> bufIn4)
        {
            result[index] = XMath.Min(XMath.Min(XMath.Min(bufIn1[index], bufIn2[index]), bufIn3[index]), bufIn4[index]);
        }

        public static void CorrelationMapToRgba32(Index2 index, ArrayView2D<uint> result, ArrayView2D<double> bufIn1)
        {
            double threshold = 0.9;
            double r = bufIn1[index];
            byte rad = 0;

            r = XMath.Abs(r);
            if (r > 1) r = 1;


            //if (r < threshold)
            //    r = 0;
            //else
            //{
            //    r = (r - threshold) * (1 / (1 - threshold));
            //    rad = (byte)(255 * XMath.Abs(r));
            //}

            rad = (byte)(255 * r);

            result[index] = (uint)(rad + (rad << 8) + (rad << 16) + (255 << 24));

            //if (r > 0)
            //    result[index] = (uint)(rad + (rad << 8) + (rad << 16) + (255 << 24));
            //else
            //    result[index] = (uint)((rad << 16) + (255 << 24));
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
                //for (int j = 0; j < 9; j++)
                //{
                //    var x = j % 9;
                //    var y = j / 9;
                //    val += (short)(kern[x, y] * sobelXKern[j]);
                //}

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
                //for (int j = 0; j < 9; j++)
                //{
                //    var x = j % 9;
                //    var y = j / 9;
                //    val += (short)(kern[x, y] * sobelYKern[x, y]);
                //}
                val += (short)(kern[0, 0] * -1);
                val += (short)(kern[1, 0] * -2);
                val += (short)(kern[2, 0] * -1);
                val += (short)(kern[0, 2] * 1);
                val += (short)(kern[1, 2] * 2);
                val += (short)(kern[2, 2] * 1);
                //result[index.X, index.Y, i] = val;
                result[index.X, index.Y, i] = kern[1, 1];
            }
        }

        public static void AccumulateSobelMap(Index3 index, ArrayView3D<short> result, ArrayView3D<short> hSobel, ArrayView3D<short> vSobel)
        {
            result[index] = KernelHelpers.Magnitude(hSobel[index], vSobel[index]);
        }

        public static void CalculateSobel(Index2 index, ArrayView2D<short> result, ArrayView3D<short> accumulatedSobel)
        {
            var val = KernelHelpers.Magnitude(accumulatedSobel.GetDepthView(index.X, index.Y));
            result[index] = val;
        }
        #endregion


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
            if (val > maxValue)
                val = maxValue;
            if (val < 0)
                val = 0;
            byte result = (byte)((val / (float)maxValue) * 255);
            output[index] = result;
        }

        public static void Max(Index index, VariableView<int> output, ArrayView<byte> input)
        {
            var compareVal = output.Value;
            var value = input[index];
            if (compareVal < input[index])
            {
                int oldVal = Atomic.CompareExchange(ref output.Value, compareVal, value);

                do
                {

                } while (false);
            }

        }

        public static void Normalize(Index index, ArrayView<float> input, float scale)
        {
            //var val = input[index];
            //input[index] = val / scale;
            input[index] /= scale;
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
        #endregion
    }
}
