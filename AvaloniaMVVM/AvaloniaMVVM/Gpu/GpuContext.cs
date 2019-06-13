using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AvaloniaMVVM.Gpu
{

    public class GpuContext : Singleton<GpuContext>, IDisposable
    {
        public Context Context { get; }
        public Accelerator Accelerator { get; }

        public GpuContext()
        {
            Context = new Context();
            Accelerator = Accelerator.Accelerators.Any(x => x.AcceleratorType == AcceleratorType.Cuda)
                ? new CudaAccelerator(Context)
                : (Accelerator)new CPUAccelerator(Context);
            //Accelerator = new CPUAccelerator(Context);
            //Accelerator = new CudaAccelerator(Context);
        }

        public void Dispose()
        {
            Accelerator?.Dispose();
            Context?.Dispose();
        }
    }

    public class Singleton<T> where T : new()
    {
        private static T instance;

        public Singleton()
        { }

        public static T Instance
        {
            get
            {
                if (instance == null)
                    instance = new T();
                return instance;
            }
        }
    }
}
