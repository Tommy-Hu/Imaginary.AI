using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    public class CrossEntropyCost : Cost
    {
        public override double GetCost(Pass pass, double[] predictedOutput)
        {
            int n = predictedOutput.Length;
            Debug.Assert(pass.expectedOutput.Length == n);

            double res = 0;
            for (int i = 0; i < n; i++)
            {
                double clippedPrediction = Math.Max(predictedOutput[i], 1e-12);
                res += pass.expectedOutput[i] * Math.Log(clippedPrediction);
            }
            return -res;
        }

        public override double GetCostDerivative(Pass pass, double[] predictedOutput, int index)
        {
            int n = predictedOutput.Length;
            Debug.Assert(pass.expectedOutput.Length == n);

            double clippedPrediction = Math.Max(predictedOutput[index], 1e-12);
            return /*1.0 / n * */-pass.expectedOutput[index] * 1.0 / clippedPrediction;
        }
    }
}
