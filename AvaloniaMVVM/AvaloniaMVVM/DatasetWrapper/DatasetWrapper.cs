using Avalonia.Media.Imaging;
using ILGPU;
using ILGPU.Runtime;
using OSGeo.GDAL;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AvaloniaMVVM.Kernels;
using System.Runtime.InteropServices;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;

namespace AvaloniaMVVM.DatasetWrapper
{
    public class DatasetWrapper : IDisposable
    {
        public static string picturePath = Path.Combine(Consts.relativePathToRoot, @"Pics/Data_Envi/samson_1.img");

        public int Width;
        public int Height;
        public int Depth;

        private Dataset dataset;
        private Context context;
        private Accelerator accelerator;
        private MemoryBuffer3D<short> dataset_v;

        public DatasetWrapper()
        {
            dataset = Gdal.Open(picturePath, Access.GA_ReadOnly);
            context = new Context();
            accelerator = Accelerator.Accelerators.Any(x => x.AcceleratorType == AcceleratorType.Cuda)
                ? new CudaAccelerator(context)
                : (Accelerator)new CPUAccelerator(context);
        }

        ~DatasetWrapper()
        {
            Dispose();
        }

        public void Dispose()
        {
            dataset?.Dispose();
            dataset_v?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();
        }

        public void LoadDatasetInVideoMemory()
        {
            int w = dataset.RasterXSize, h = dataset.RasterYSize, d = dataset.RasterCount;
            Width = w; Height = h; Depth = d;

            short[] buffer = new short[w * h * d];
            var bands = Enumerable.Range(1, d).ToArray();
            dataset.ReadRaster(0, 0, w, h, buffer, w, h, d, bands, 0, 0, 0);
            dataset_v = accelerator.Allocate<short>(w, h, d);
            dataset_v.CopyFrom(buffer, 0, Index3.Zero, buffer.Length);
        }

        public void RenderBand(ref WriteableBitmap bmp, int band)
        {
            int w = dataset.RasterXSize, h = dataset.RasterYSize;
            var pinConvertKernel = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView<uint>, ArrayView<short>, double, short>(Kernels.Kernels.PicConvertion);
            // todo: MinMax should be calculated on GPU
            double[] MinMax = new double[2];
            dataset.GetRasterBand(band).ComputeRasterMinMax(MinMax, 0);
            short min = (short)(MinMax[0]);
            double mult = 255 / (MinMax[1] - MinMax[0]);

            uint[] data;
            using (var bufOut = accelerator.Allocate<uint>(w * h))
            {
                pinConvertKernel(bufOut.Length, bufOut.View, dataset_v.GetSliceView(band - 1).AsLinearView(), mult, min);
                accelerator.Synchronize();
                data = bufOut.GetAsArray();
            }
            using (var buf = bmp.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
        }
    }
}
