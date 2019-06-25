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

    public class GpuContext : Singleton<GpuContext>
    {
        public Context Context { get; }
        public Accelerator Accelerator { get; }

        public GpuContext()
        {
            Context = new Context();
            Accelerator = Accelerator.Accelerators.Any(x => x.AcceleratorType == AcceleratorType.Cuda)
                ? new CudaAccelerator(Context)
                : (Accelerator)new CPUAccelerator(Context);
        }
        
        ~GpuContext()
        {
            Accelerator?.Dispose();
            Context?.Dispose();
        }
    }

    public class Singleton<T> where T : new()
    {
        private static T _instance;

        public Singleton() { }

        public static T Instance
        {
            get
            {
                if (_instance == null)
                    _instance = new T();
                return _instance;
            }
        }
    }
}
