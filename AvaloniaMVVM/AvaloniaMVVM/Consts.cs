using System.IO;
using System.Reflection;

namespace AvaloniaMVVM
{
    public static class Consts
    {
        public static readonly string RelativePathToRoot = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), @"../../../");
    }
}
