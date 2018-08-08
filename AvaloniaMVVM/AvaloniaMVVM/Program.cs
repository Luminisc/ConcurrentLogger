using System;
using Avalonia;
using Avalonia.Logging.Serilog;
using AvaloniaMVVM.ViewModels;
using AvaloniaMVVM.Views;
using System.Runtime.InteropServices;
using OSGeo.GDAL;
using System.IO;

namespace AvaloniaMVVM
{
    class Program
    {
        public static readonly string relativePathToRoot = @"..\..\..\..\";
        public static string picturePath = Path.Combine(relativePathToRoot, @"Pics\Data_Envi\samson_1.img");
        
        static void Main(string[] args)
        {
            Dataset dataset;
            double x = Add(5.0, 17.0);
            Console.WriteLine($"5 + 17 = {x}");
            Console.WriteLine($"Current folder: {Environment.CurrentDirectory}");
            Gdal.AllRegister();
            dataset = Gdal.Open(picturePath, Access.GA_ReadOnly);
            Console.WriteLine($"Dataset loaded successfully.");
            Console.WriteLine($"Dataset dimensions:\r\n\tWidth: {dataset.RasterXSize}\r\n\tHeight: {dataset.RasterYSize}\r\n\tBands: {dataset.RasterCount}");
            BuildAvaloniaApp().Start<MainWindow>(() => new MainWindowViewModel());
        }

        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .UseReactiveUI()
                .LogToDebug();

        //#if OS_LINUX
        //		[DllImport(@"./CMakeLibrary.so", EntryPoint = "Add")]
        //#endif
        //#if OS_WINDOWS
        //      [DllImport(@"./CMakeLibrary.dll", EntryPoint = "Add")]
        //#endif
        [DllImport(@"./CMakeLibrary", EntryPoint = "Add")]
        public static extern double Add(double x, double y);
    }
}
