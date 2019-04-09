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
            var vm = new MainWindowViewModel();
            Stopwatch st = new Stopwatch();
            st.Start();
            Dataset dataset = Gdal.Open(picturePath, Access.GA_ReadOnly);
            st.Stop();
            Console.WriteLine($"Dataset loaded successfully in {st.ElapsedMilliseconds} ms.");
            Console.WriteLine($"Dataset dimensions:\r\n\tWidth: {dataset.RasterXSize}\r\n\tHeight: {dataset.RasterYSize}\r\n\tBands: {dataset.RasterCount}");
            PixelSize ps = new PixelSize(dataset.RasterXSize, dataset.RasterYSize);
            Vector dpi = new Vector(1, 1);
            WriteableBitmap bmp = new WriteableBitmap(ps, dpi, Avalonia.Platform.PixelFormat.Rgba8888);
            short[] buffer;
            double mult = 0;
            short min = 0;
            st.Restart();
            using (var buf = bmp.Lock())
            {
                Console.WriteLine($"Bmp rowBytes: {buf.RowBytes}");
                IntPtr ptr = buf.Address;
                var band = dataset.GetRasterBand(5);
                buffer = new short[dataset.RasterXSize * dataset.RasterYSize];
                band.ReadRaster(0, 0, dataset.RasterXSize, dataset.RasterYSize, buffer, dataset.RasterXSize, dataset.RasterYSize, 0, 0);
                double[] args = new double[2];
                band.ComputeRasterMinMax(args, 0);
                Console.WriteLine($"Min: {args[0]}, Max: {args[1]}");
                min = (short)(args[0]);
                mult = 255 / (args[1] - args[0]);
                var rgbaBuf = buffer
                    .Select(x => (byte)((x - min) * mult))
                    .SelectMany(x => new byte[] { x, x, x, 255 })
                    .ToArray();
                Marshal.Copy(rgbaBuf, 0, ptr, rgbaBuf.Length);
            }
            //bmp.Save("D:/pic.png");
            st.Stop();
            Console.WriteLine($"Elapsed: {st.ElapsedMilliseconds}");
            vm.Greeting = "ololo";
            vm.RenderImage = bmp;

            var data = new uint[0];
            foreach (var acc in Accelerator.Accelerators)
            {
                Console.WriteLine($"{acc.AcceleratorType} {acc.DeviceId}");
            }

            using (var context = new Context())
            {
                using (var accelerator = new CudaAccelerator(context))
                {
                    var myKernel = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<uint>, ArrayView<short>, double, short>(PicConvertion);

                    // Allocate some memory
                    using (var bufOut = accelerator.Allocate<uint>(dataset.RasterXSize * dataset.RasterYSize))
                    using (var bufIn = accelerator.Allocate<short>(dataset.RasterXSize * dataset.RasterYSize))
                    {
                        bufIn.CopyFrom(buffer, 0, 0, buffer.Length);
                        // Launch buffer.Length many threads and pass a view to buffer
                        myKernel(buffer.Length, bufOut.View, bufIn.View, mult, min);

                        // Wait for the kernel to finish...
                        accelerator.Synchronize();

                        // Resolve data
                        data = bufOut.GetAsArray();
                    }
                }
            }

            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
            dataset.Dispose();
            return vm;
        }

        static void MyKernel(
            Index index, // The global thread index (1D in this case)
            ArrayView<int> dataView, // A view to a chunk of memory (1D in this case)
        int constant) // A sample uniform constant
        {
            dataView[index] = index + constant;
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
