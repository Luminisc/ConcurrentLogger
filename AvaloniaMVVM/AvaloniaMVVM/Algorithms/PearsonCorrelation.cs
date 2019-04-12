using System;
using System.Linq;

namespace AvaloniaMVVM.Algorithms
{
    public class PearsonCorrelation
    {
        public double Calculate(short[] xs, short[] ys)
        {
            double sx = 0.0;
            double sy = 0.0;
            double sxx = 0.0;
            double syy = 0.0;
            double sxy = 0.0;

            int n = 0;

            var minLength = Math.Min(xs.Length, ys.Length);
            for (int i = 0; i < minLength; i++)
            {
                short x = xs[i];
                short y = ys[i];

                n += 1;
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }

            double covariation = sxy / n - sx * sy / n / n;
            double sigmaX = Math.Sqrt(sxx / n - sx * sx / n / n);
            double sigmaY = Math.Sqrt(syy / n - sy * sy / n / n);
            
            return covariation / (sigmaX * sigmaY);
        }
    }
}
