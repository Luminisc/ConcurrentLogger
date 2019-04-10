using System;
using System.Collections.Generic;
using System.Text;
using Avalonia;
using Avalonia.Media.Imaging;
using ReactiveUI;
using AvaloniaMVVM.DatasetWrapper;

namespace AvaloniaMVVM.ViewModels
{
    public class MainWindowViewModel : ViewModelBase, IDisposable
    {
        public MainWindowViewModel()
        {
            wrapper.LoadDatasetInVideoMemory();
            RenderImage = new WriteableBitmap(new PixelSize(wrapper.Width, wrapper.Height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888);
        }

        public string Greeting { get; set; } = "Hello World!";

        private WriteableBitmap _renderImage;
        public WriteableBitmap RenderImage
        {
            get => _renderImage;
            set => this.RaiseAndSetIfChanged(ref _renderImage, value);
        }
        
        protected int _band = 1;
        public int Band
        {
            get => _band;
            set => this.RaiseAndSetIfChanged(ref _band, value);
        }

        protected DatasetWrapper.DatasetWrapper wrapper = new DatasetWrapper.DatasetWrapper();

        public void ChangeBand()
        {
            var band = Band;
            if (band < 1) band = 1;
            if (band > wrapper.Depth) band = wrapper.Depth;
            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderBand(ref img, band);
            RenderImage = img;
            
            Band = band;
        }

        public void Dispose()
        {
            wrapper.Dispose();
        }
    }
}
