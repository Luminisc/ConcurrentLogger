using System;
using Avalonia;
using Avalonia.Logging.Serilog;
using AvaloniaMVVM.ViewModels;
using AvaloniaMVVM.Views;
using OSGeo.GDAL;
using System.IO;
using System.Reflection;
using Directory = System.IO.Directory;

namespace AvaloniaMVVM
{
    internal class Program
    {
        public static readonly string RelativePathToRoot = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), "../../../");
        public static readonly string PathToTemp = Path.Combine(RelativePathToRoot, "/Temp");

        private static void Main(string[] args)
        {
            Console.WriteLine($"Current folder: {RelativePathToRoot}");
            Console.Write("Registering GDAL... ");
            Gdal.AllRegister();
            Console.WriteLine("Successful.");

            if (!Directory.Exists(PathToTemp))
            {
                Directory.CreateDirectory(PathToTemp);
            }

            BuildAvaloniaApp()
                .Start<MainWindow>(GetModel);
            Console.WriteLine("Exitting...");
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

        //[DllImport(@"./CMakeLibrary", EntryPoint = "Add")]
        //public static extern double Add(double x, double y);
    }
}
