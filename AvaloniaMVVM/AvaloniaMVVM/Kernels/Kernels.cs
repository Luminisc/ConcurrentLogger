using ILGPU;
using System;
using System.Collections.Generic;
using System.Text;

namespace AvaloniaMVVM.Kernels
{
    public class Kernels
    {
        public static void PicConvertion(Index index, ArrayView<uint> buf1, ArrayView<short> buf2, double mult, short min)
        {
            byte rad = (byte)((buf2[index] - min) * mult);
            buf1[index] = (uint)(rad + (rad << 8) + (rad << 16) + (255 << 24));
        }


    }
}
