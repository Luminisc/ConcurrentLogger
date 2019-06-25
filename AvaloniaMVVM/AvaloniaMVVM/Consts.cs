using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;

namespace AvaloniaMVVM
{
    public static class Consts
    {
        public static readonly string RelativePathToRoot = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), @"../../../");
    }
}
