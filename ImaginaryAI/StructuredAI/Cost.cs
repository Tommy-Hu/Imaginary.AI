using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    public abstract class Cost
    {
        public abstract double GetCost(Pass pass, double[] predicted);

        /// <summary>
        /// Calculates the <paramref name="index"/>'th partial derivative of the cost function.
        /// The result should be used for the <paramref name="index"/>'th neuron's back propagation.
        /// </summary>
        /// <param name="pass"></param>
        /// <param name="predicted"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public abstract double GetCostDerivative(Pass pass, double[] predicted, int index);
    }
}
