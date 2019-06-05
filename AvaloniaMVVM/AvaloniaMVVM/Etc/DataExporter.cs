using AvaloniaMVVM.Kernels;
using System.IO;
using System;
using System.Collections.Generic;
using System.Text;

namespace AvaloniaMVVM.Etc
{
    public class DataExporter
    {
        public static void ExportHistogramInCsv(HistogramData data)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Brightness;Count");
            for (int i = 0; i < data.histogramData.Length; i++)
            {
                sb.AppendLine($"{i + 1};{data.histogramData[i]}");
            }
            File.WriteAllText("histogramData.csv", sb.ToString());
        }

        public static void ExportBrightnessInCsv(BrightnessCalculationData data)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Channel;Mean value;Max value; Deviation value");
            for (int i = 0; i < data.arrMeanBrightness.Length; i++)
            {
                sb.AppendLine($"{i + 1};{data.arrMeanBrightness[i]};{data.arrMaxBrightness[i]};{data.arrStandartDeviation[i]}");
            }
            File.WriteAllText("brightnessData.csv", sb.ToString());
        }
    }
}
