using System;
using System.Collections.Generic;
using System.Text;

namespace AvaloniaMVVM.Kernels
{
    public class ColorConsts
    {
        public static readonly int Black = 0 + (255 << 24);
        public static readonly int White = -1;
        public static readonly int Red = 255 + (255 << 24);
        public static readonly int Green = (255 << 8) + (255 << 24);
        public static readonly int Blue = (255 << 16) + (255 << 24);
    }
}
