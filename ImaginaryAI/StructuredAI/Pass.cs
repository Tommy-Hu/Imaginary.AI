using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    /// <summary>
    /// Represents a pass in training.
    /// </summary>
    public struct Pass
    {
        public double[] inputs;
        public double[] expectedOutput;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public Pass(double[] inputs, double[] expectedOutput)
        {
            this.inputs = inputs;
            this.expectedOutput = expectedOutput;
        }

        public int ExpectedClass
        {
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            get
            {
                double maxSoFar = double.NegativeInfinity;
                int maxInd = 0;
                for (int i = 0; i < expectedOutput.Length; i++)
                {
                    if (expectedOutput[i] > maxSoFar)
                    {
                        maxSoFar = expectedOutput[i];
                        maxInd = i;
                    }
                }
                return maxInd;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public Pass Clone()
        {
            return new Pass((double[])inputs.Clone(), (double[])expectedOutput.Clone());
        }
    }
}
