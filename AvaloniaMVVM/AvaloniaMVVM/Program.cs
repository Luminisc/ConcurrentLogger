using System;
using Avalonia;
using Avalonia.Logging.Serilog;
using AvaloniaMVVM.ViewModels;
using AvaloniaMVVM.Views;
using System.Runtime.InteropServices;
using OSGeo.GDAL;
using System.Linq;
using System.IO;
using System.Reflection;
using Avalonia.Media.Imaging;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

namespace AvaloniaMVVM
{
    class Program
    {
        public static readonly string relativePathToRoot = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), @"../../../");
        public static string picturePath = Path.Combine(relativePathToRoot, @"Pics/Data_Envi/samson_1.img");

        static void Main(string[] args)
        {
            Console.WriteLine($"Current folder: {relativePathToRoot}");
            Console.WriteLine();
            Gdal.AllRegister();

            BuildAvaloniaApp().Start<MainWindow>(() => GetModel());
        }

        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .UseReactiveUI()
                .LogToDebug();

        public static MainWindowViewModel GetModel()
        {
            return new MainWindowViewModel();
        }

        static void PicConvertion(Index index, ArrayView<uint> buf1, ArrayView<short> buf2, double mult, short min)
        {
            byte rad = (byte)((buf2[index] - min) * mult);
            buf1[index] = (uint)(rad + (rad << 8) + (rad << 16) + (255 << 24));
        }

        //[DllImport(@"./CMakeLibrary", EntryPoint = "Add")]
        //public static extern double Add(double x, double y);
    }
}
