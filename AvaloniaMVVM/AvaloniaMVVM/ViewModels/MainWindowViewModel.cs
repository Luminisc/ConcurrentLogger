using System;
using System.Collections.Generic;
using System.Text;
using Avalonia;
using Avalonia.Media.Imaging;
using ReactiveUI;
using AvaloniaMVVM.DatasetWrapper;
using Avalonia.Input;
using System.Linq;
using AvaloniaMVVM.Etc;

namespace AvaloniaMVVM.ViewModels
{
    public class MainWindowViewModel : ViewModelBase, IDisposable
    {

        public MainWindowViewModel()
        {

        }

        public string Greeting { get; set; } = "Hello World!";

        private string _description;
        public string Description
        {
            get => _description;
            set => this.RaiseAndSetIfChanged(ref _description, value);
        }

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
            set
            {
                var val = value;
                if (wrapper != null)
                    val = Math.Clamp(val, 0, wrapper.Depth);

                if (_band != val)
                    ChangeBand();

                this.RaiseAndSetIfChanged(ref _band, val);
            }
        }

        protected DatasetWrapper.DatasetWrapper wrapper;
        protected bool loaded = false;
        protected double maxMeanBrightness = 0.0;

        public void ChangeBand()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderBand(ref img, Band);
            RenderImage = img;
        }

        public void RenderCorrelation()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderCorrelation(ref img);
            RenderImage = img;
        }

        public void RenderSobelMap()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderSobel(ref img);
            RenderImage = img;
        }

        public void RenderHistogram()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            var hist = wrapper.RenderHistogram(ref img);
            RenderImage = img;

            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Maximum brightness: {hist.histogramData.Length}");
            sb.AppendLine($"Most frequent brightness: {hist.histogramData.Skip(1).Max()}");
            Description = sb.ToString();

            DataExporter.ExportHistogramInCsv(hist);
        }

        public void RenderBrightnessCalculationData()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            var calcs = wrapper.GetBrightnessCalculationData(ref img);
            RenderImage = img;

            StringBuilder sb = new StringBuilder();
            sb.AppendLine("Red: Mean");
            sb.AppendLine("Green: Max");
            sb.AppendLine("Blue: Deviation");
            sb.AppendLine("");
            var maxMean = calcs.arrMeanBrightness.Max();
            maxMeanBrightness = maxMean;
            var maxMeanBand = calcs.arrMeanBrightness.ToList().IndexOf(maxMean);
            sb.AppendLine($"Maximum mean value: {maxMean} in {maxMeanBand + 1} band");

            Description = sb.ToString();
            DataExporter.ExportBrightnessInCsv(calcs);
        }

        public void ConvertToByteRepresentation()
        {
            var img = _renderImage;
            RenderImage = null;
            wrapper.ConvertToByteRepresentation(ref img, (short)(maxMeanBrightness * 2));
            RenderImage = img;
        }

        public void RenderPearsonCorrelation()
        {
            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderPearsonCorrelation(ref img);
            RenderImage = img;
        }

        public void RunWorkflow()
        {
            RenderBrightnessCalculationData();
            ConvertToByteRepresentation();
            RenderPearsonCorrelation();
        }

        public void SaveImage()
        {
            _renderImage.Save($"D://img_{DateTime.Now.ToFileTime()}.png");
        }

        public void InitializeDataset()
        {
            if (loaded)
                return;

            wrapper = new DatasetWrapper.DatasetWrapper();
            RenderImage = new WriteableBitmap(new PixelSize(wrapper.Width, wrapper.Height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888);
            wrapper.LoadDatasetInVideoMemory();
            loaded = true;
        }

        public void Dispose()
        {
            wrapper?.Dispose();
        }

        public void OnImagePress(Point clickPoint, Size controlSize)
        {

        }
    }
}
