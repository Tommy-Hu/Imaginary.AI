using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    public class QuadraticCost : Cost
    {
        public override double GetCost(Pass pass, double[] predicted)
        {
            int n = predicted.Length;
            double acc = 0;
            for (int i = 0; i < n; i++)
            {
                // (p - y) ^ 2
                double lambda = (predicted[i] - pass.expectedOutput[i]) * (predicted[i] - pass.expectedOutput[i]);
                acc += lambda;
            }
            return acc;
        }

        public override double GetCostDerivative(Pass pass, double[] predicted, int index)
        {
            // 2 * (p - y)
            return 2.0 * (predicted[index] - pass.expectedOutput[index]);
        }
    }
}
