using ILGPU;
using System.Linq;

namespace AvaloniaMVVM.Kernels
{
    internal static class KernelExtentions
    {
        //public static void Multiply<T>(this ArrayView<T> arr, Index index, T mult) where T : struct
        //{

        //}

        //public static ArrayView<T> GetDepthView<T>(this ArrayView3D<T> view, int x, int y) where T : struct
        //{
        //    return view.GetSubView(new Index3(x, y, 0), new Index3(1, 1, view.Depth)).AsLinearView();
        //}
    }

    internal static class KernelHelpers
    {
        public static short Magnitude(short x, short y)
        {
            return (short)XMath.Sqrt(XMath.Pow(x, 2) + XMath.Pow(y, 2));
        }

        public static short Magnitude(params short[] a)
        {
            int sum = a.Sum(x => (int)XMath.Pow(x, 2));
            return (short)XMath.Sqrt(sum);
        }

        public static short Magnitude(ArrayView<short> a)
        {
            var extent = a.Extent.X;
            int sum = 0;
            for (int i = 0; i < extent; i++)
            {
                sum = (int)XMath.Pow(a[i], 2);
            }
            return (short)XMath.Sqrt(sum);
        }

        public static short Magnitude(ArrayView3D<short> a, Index2 index)
        {
            var extent = a.Depth;
            int sum = 0;
            for (int i = 0; i < extent; i++)
            {
                sum = (int)XMath.Pow(a[index.X, index.Y, i], 2);
            }
            return (short)XMath.Sqrt(sum);
        }

        public static int FindLastNotZeroIndex(int[] arr)
        {
            for (int i = arr.Length - 1; i > 0; i--)
            {
                if (arr[i] != 0) return i;
            }
            return 0;
        }
    }
}
