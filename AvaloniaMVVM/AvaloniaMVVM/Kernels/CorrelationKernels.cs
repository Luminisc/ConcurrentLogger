using ILGPU;
using System;
using System.Collections.Generic;
using System.Text;

namespace AvaloniaMVVM.Kernels
{
    public class CorrelationKernels
    {
        public static void CorrelationMap(Index2 index, ArrayView3D<float> result, ArrayView3D<byte> bufIn)
        {
            if (index.X == bufIn.Width - 1
                || index.X == 0
                || index.Y == bufIn.Height - 1
                || index.Y == 0)
            {
                result[index.X, index.Y, 0] = 1.0f;
                result[index.X, index.Y, 1] = 1.0f;
                result[index.X, index.Y, 2] = 1.0f;
                return;
            }
            
            var corrX = (float)XMath.Abs(Kernels.PearsonCorrelation(bufIn, new Index2(index.X - 1, index.Y), new Index2(index.X + 1, index.Y)));
            var corrY = (float)XMath.Abs(Kernels.PearsonCorrelation(bufIn, new Index2(index.X, index.Y - 1), new Index2(index.X, index.Y + 1)));
            
            result[index.X, index.Y, 0] = (float)((1.0f - corrX));
            result[index.X, index.Y, 1] = (float)(XMath.Sqrt(XMath.Pow(1.0f - corrX, 2) + XMath.Pow(1.0f - corrY, 2)));
            result[index.X, index.Y, 2] = (float)((1.0f - XMath.Min(corrX, corrY)));
        }

        public static void CorrelationSobelMap(Index2 index, ArrayView3D<float> result, ArrayView3D<byte> bufIn)
        {
            if (index.X == bufIn.Width - 1
                || index.X == 0
                || index.Y == bufIn.Height - 1
                || index.Y == 0)
            {
                result[index.X, index.Y, 0] = 1.0f;
                result[index.X, index.Y, 1] = 1.0f;
                result[index.X, index.Y, 2] = 1.0f;
                return;
            }

            var corrX = (float)XMath.Abs(Kernels.PearsonCorrelation(bufIn, new Index2(index.X - 1, index.Y), new Index2(index.X + 1, index.Y)));
            var corrY = (float)XMath.Abs(Kernels.PearsonCorrelation(bufIn, new Index2(index.X, index.Y - 1), new Index2(index.X, index.Y + 1)));

            result[index.X, index.Y, 0] = (float)((1.0f - corrX));
            result[index.X, index.Y, 1] = (float)(XMath.Sqrt(XMath.Pow(1.0f - corrX, 2) + XMath.Pow(1.0f - corrY, 2)));
            result[index.X, index.Y, 2] = (float)((1.0f - XMath.Min(corrX, corrY)));
        }
    }
}
