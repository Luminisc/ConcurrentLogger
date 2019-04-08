using System;
using System.Collections.Generic;
using System.Text;
using Avalonia;
using Avalonia.Media.Imaging;

namespace AvaloniaMVVM.ViewModels
{
    public class MainWindowViewModel : ViewModelBase
    {
        public string Greeting { get; set; } = "Hello World!";

        public WriteableBitmap RenderImage;
    }
}
