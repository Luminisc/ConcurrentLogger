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
            //Dataset dataset = Gdal.Open(picturePath, Access.GA_ReadOnly);
            //Console.WriteLine($"Dataset loaded successfully.");
            //Console.WriteLine($"Dataset dimensions:\r\n\tWidth: {dataset.RasterXSize}\r\n\tHeight: {dataset.RasterYSize}\r\n\tBands: {dataset.RasterCount}");
            //var band = dataset.GetRasterBand(1);
            //var buffer = new short[dataset.RasterXSize * dataset.RasterYSize];
            //band.ReadRaster(0, 0, dataset.RasterXSize, dataset.RasterYSize, buffer, dataset.RasterXSize, dataset.RasterYSize, 0, 0);
            //var signature = new short[dataset.RasterCount];
            //dataset.ReadRaster(0, 0, 1, 1, signature, 1, 1, dataset.RasterCount, Enumerable.Range(1, dataset.RasterCount).ToArray(), 0, 0, 0);
            //Console.WriteLine($"Buffer: {string.Join(", ", buffer.Take(10))} ...");
            //Console.WriteLine($"Signature: {string.Join(", ", signature.Take(10))} ...");
            BuildAvaloniaApp().Start<MainWindow>(() => GetModel());
        }

        public static AppBuilder BuildAvaloniaApp()
            => AppBuilder.Configure<App>()
                .UsePlatformDetect()
                .UseReactiveUI()
                .LogToDebug();

        public static MainWindowViewModel GetModel()
        {
            var vm = new MainWindowViewModel();
            Stopwatch st = new Stopwatch();

            Dataset dataset = Gdal.Open(picturePath, Access.GA_ReadOnly);
            Console.WriteLine($"Dataset loaded successfully.");
            Console.WriteLine($"Dataset dimensions:\r\n\tWidth: {dataset.RasterXSize}\r\n\tHeight: {dataset.RasterYSize}\r\n\tBands: {dataset.RasterCount}");
            PixelSize ps = new PixelSize(dataset.RasterXSize, dataset.RasterYSize);
            Vector dpi = new Vector(1,1);
            WriteableBitmap bmp = new WriteableBitmap(ps, dpi, Avalonia.Platform.PixelFormat.Rgba8888);
            
            st.Start();
            using (var buf = bmp.Lock())
            {
                Console.WriteLine($"Bmp rowBytes: {buf.RowBytes}");
                IntPtr ptr = buf.Address;
                var band = dataset.GetRasterBand(5);
                var buffer = new short[dataset.RasterXSize * dataset.RasterYSize];
                band.ReadRaster(0, 0, dataset.RasterXSize, dataset.RasterYSize, buffer, dataset.RasterXSize, dataset.RasterYSize, 0, 0);
                double[] args = new double[2];
                band.ComputeRasterMinMax(args, 0);
                Console.WriteLine($"Min: {args[0]}, Max: {args[1]}");
                var mult = 255 / args[1] - args[0];
                var rgbaBuf = buffer
                    .Select(x => (byte)(x * mult))
                    .SelectMany(x => new byte[] { x, Math.Max((byte)60, x), 0, 255 })
                    .ToArray();
                Marshal.Copy(rgbaBuf, 0, ptr, rgbaBuf.Length);
            }
            //bmp.Save("D:/pic.png");
            st.Stop();
            Console.WriteLine($"Elapsed: {st.ElapsedMilliseconds}");
            vm.Greeting = "ololo";
            vm.RenderImage = bmp;
            dataset.Dispose();
            return vm;
        }

        //[DllImport(@"./CMakeLibrary", EntryPoint = "Add")]
        //public static extern double Add(double x, double y);
    }
}
