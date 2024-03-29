﻿using ILGPU;

namespace AvaloniaMVVM.Kernels
{
    public struct BrightnessCalculationData
    {
        public int[] arrImage;
        public double[] arrMeanBrightness;
        public short[] arrMaxBrightness;
        public double[] arrStandardDeviation;
        public Index2 imageSize;
    }

    public class HistogramData
    {
        public int[] histogramData;
    }

    public class CorrelationData
    {
        public uint[] xCorrelation;
        public uint[] yCorrelation;
        public uint[] xyCorrelation;
        public byte[] rawPicture;
    }

    public class PicturesData
    {
        public uint[] xPicture;
        public uint[] yPicture;
        public uint[] xyPicture;
        public byte[] rawPicture;
    }

    public class AccumulationData
    {

    }
}
