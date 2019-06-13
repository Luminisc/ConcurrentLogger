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
using System.Runtime.InteropServices;

namespace AvaloniaMVVM.ViewModels
{
    public class MainWindowViewModel : ViewModelBase, IDisposable
    {

        public MainWindowViewModel()
        {

        }

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

        protected byte lowThresholdValue = 0;
        public byte LowThresholdValue
        {
            get => lowThresholdValue;
            set
            {
                var val = value;
                this.RaiseAndSetIfChanged(ref lowThresholdValue, val);
            }
        }
        protected byte highThresholdValue = 0;
        public byte HighThresholdValue
        {
            get => highThresholdValue;
            set
            {
                var val = value;
                this.RaiseAndSetIfChanged(ref highThresholdValue, val);
            }
        }


        protected DatasetWrapper.DatasetWrapper wrapper;
        protected bool loaded = false;
        protected double maxMeanBrightness = short.MaxValue / 2;

        public void ChangeBand()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderBand(ref img, Band);
            RenderImage = img;
        }

        public void RenderCanny()
        {
            InitializeDataset();

            if (_renderImage == null)
                return;

            var rimg = _renderImage;

            SaveImage("Precanny.png");
            OpenCvSharp.Mat src = new OpenCvSharp.Mat("Precanny.png", OpenCvSharp.ImreadModes.Grayscale);
            OpenCvSharp.Mat dst = new OpenCvSharp.Mat();
            OpenCvSharp.Cv2.Canny(src, dst, LowThresholdValue, HighThresholdValue);

            byte[] img = new byte[wrapper.Width * wrapper.Height];
            dst.GetArray(0, 0, img);
            uint[] data = img.Select(x => x == 0 ? (uint)0 + 255 << 24 : uint.MaxValue).ToArray();

            using (var buf = rimg.Lock())
            {
                IntPtr ptr = buf.Address;
                Marshal.Copy((int[])(object)data, 0, ptr, data.Length);
            }
            RenderImage = null;
            RenderImage = rimg;
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

            DataExporter.ExportHistogramInCsv(hist, wrapper.Depth);
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
            DataExporter.ExportBrightnessInCsv(calcs, wrapper.Depth);
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
            var img = new WriteableBitmap(new PixelSize(wrapper.Width, wrapper.Height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888);
            //RenderImage = null;
            wrapper.RenderPearsonCorrelation(ref img, 0, highThresholdValue);
            RenderImage = img;
        }

        public void RenderSignatureLengthDerivative()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderSignatureLengthDerivative(ref img, false, (short)(maxMeanBrightness * 2), highThresholdValue);
            RenderImage = img;
        }

        public void RenderSignatureLengthDerivativeNormalize()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderSignatureLengthDerivative(ref img, true, (short)(maxMeanBrightness * 2), highThresholdValue);
            RenderImage = img;
        }

        public void RenderByteSignatureLengthDerivative()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderByteSignatureLengthDerivative(ref img, false);
            RenderImage = img;
        }

        public void RenderByteSignatureLengthDerivativeNormalize()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderByteSignatureLengthDerivative(ref img, true);
            RenderImage = img;
        }

        public void RunWorkflow()
        {
            RenderBrightnessCalculationData();
            ConvertToByteRepresentation();
            RenderPearsonCorrelation();
        }

        public void SaveImage(string path = "")
        {
            if (string.IsNullOrWhiteSpace(path))
                _renderImage.Save($"D://img_{DateTime.Now.ToFileTime()}.png");
            else
                _renderImage.Save(path);
        }

        public void AccumulateEdges()
        {
            InitializeDataset();

            // canny
            var old = _renderImage;
            _renderImage = new WriteableBitmap(new PixelSize(wrapper.Width, wrapper.Height), new Vector(1, 1), Avalonia.Platform.PixelFormat.Rgba8888);
            wrapper.RenderBand(ref _renderImage, 17); //Because 17th band is close to green spectrum

            SaveImage("Precanny.png");
            OpenCvSharp.Mat src = new OpenCvSharp.Mat("Precanny.png", OpenCvSharp.ImreadModes.Grayscale);
            OpenCvSharp.Mat dst = new OpenCvSharp.Mat();
            OpenCvSharp.Cv2.Canny(src, dst, lowThresholdValue, 100);    // 0 and 100 thresholds giving more or less clear picture of edges

            byte[] cannyData = new byte[wrapper.Width * wrapper.Height];
            dst.GetArray(0, 0, cannyData);

            var img = _renderImage;
            wrapper.AccumulateEdges(ref img, cannyData, HighThresholdValue, (short)(maxMeanBrightness * 2));
            RenderImage = null;
            RenderImage = img;

            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"White: Pearson correlation with threshold = {HighThresholdValue}");
            sb.AppendLine("Red: Canny edge detection");
            sb.AppendLine("Green: Normalized signatures difference length - byte representation");
            sb.AppendLine("Blue: Normalized signatures difference length - short representation");

            Description = sb.ToString();

            SaveImage($"D://Temp/accumulatedEdges_{wrapper.Depth}Bands_{HighThresholdValue}Threshold.png");
        }

        public void RenderPseudoColor()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderPseudoColor(ref img, (short)(maxMeanBrightness * 2));
            RenderImage = img;
        }

        public void RenderScanline()
        {
            InitializeDataset();

            var img = _renderImage;
            RenderImage = null;
            wrapper.RenderScanline(ref img, Band-1, LowThresholdValue, HighThresholdValue);
            RenderImage = img;
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
