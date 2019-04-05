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

namespace AvaloniaMVVM
{
    class Program
    {
        public static readonly string relativePathToRoot = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), @"../../../");
        public static string picturePath = Path.Combine(relativePathToRoot, @"Pics/Data_Envi/samson_1.img");

        static void Main(string[] args)
        {
            //double x = Add(5.0, 17.0);
            //Console.WriteLine($"5 + 17 = {x}");
            Console.WriteLine($"Current folder: {relativePathToRoot}");
            Console.WriteLine();
            Gdal.AllRegister();
            Dataset dataset = Gdal.Open(picturePath, Access.GA_ReadOnly);
            Console.WriteLine($"Dataset loaded successfully.");
            Console.WriteLine($"Dataset dimensions:\r\n\tWidth: {dataset.RasterXSize}\r\n\tHeight: {dataset.RasterYSize}\r\n\tBands: {dataset.RasterCount}");
            var band = dataset.GetRasterBand(1);
            var buffer = new byte[95 * 95];
            band.ReadRaster(0, 0, 95, 95, buffer, 95, 95, 0, 0);
            Console.WriteLine($"Buffer: {string.Join(", ", buffer.Take(10))} ...");
            BuildAvaloniaApp().Start<MainWindow>(() => new MainWindowViewModel());
        }

        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .UseReactiveUI()
                .LogToDebug();

        //[DllImport(@"./CMakeLibrary", EntryPoint = "Add")]
        //public static extern double Add(double x, double y);
    }
}
