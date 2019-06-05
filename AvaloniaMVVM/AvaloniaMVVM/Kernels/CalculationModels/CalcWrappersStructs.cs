﻿using ILGPU;
using System;
using System.Collections.Generic;
using System.Text;

namespace AvaloniaMVVM.Kernels
{
    public struct BrightnessCalculationData
    {
        public int[] arrImage;
        public double[] arrMeanBrightness;
        public short[] arrMaxBrightness;
        public double[] arrStandartDeviation;
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
    }
}