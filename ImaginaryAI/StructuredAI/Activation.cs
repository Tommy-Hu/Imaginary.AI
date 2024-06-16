using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    public class Activation
    {
        public readonly Func<double, double> activationFunc;
        public readonly Func<double, double> activationDerivative;


        private static Activation? sigmoid = null;
        public static Activation Sigmoid => sigmoid ?? new Activation(A_Sigmoid, AP_SigmoidDerivative);
        private static Activation? error = null;
        public static Activation Error => new Activation(_Error, _Error);

        private Activation(Func<double, double> activationFunc, Func<double, double> activationDerivative)
        {
            this.activationFunc = activationFunc;
            this.activationDerivative = activationDerivative;
        }

        private static double _Error(double neuronValue)
        {
            throw new ImaginaryException("Error!");
        }

        /// <summary>
        /// This function is also known as the a() function.
        /// </summary>
        /// <param name="neuronValue">Also known as the z value.</param>
        /// <returns></returns>
        private static double A_Sigmoid(double neuronValue)
        {
            // here, I use the sigmoid / logistic function, but can also use other functions like tanh
            return 1.0 / (1.0 + Math.Exp(-neuronValue));
        }

        /// <summary>
        /// Finds a'(z) where z is the cached neuron value.
        /// </summary>
        /// <returns></returns>
        private static double AP_SigmoidDerivative(double neuronValue)
        {
            // the derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
            var sx = A_Sigmoid(neuronValue);
            return sx * (1.0 - sx);
        }

        private static double A_Softmax(double[] neuronValues, int ind)
        {
            // neuronValues is an array that represents the output vector.
            if (neuronValues.Length == 0) return -1.0;// something went wrong

            double tot = 0.0;
            double[] exp = new double[neuronValues.Length];
            double max = neuronValues.Max();
            for (int i = 0; i < neuronValues.Length; i++)
            {
                exp[i] = Math.Exp(neuronValues[i] - max);
                tot += exp[i];
            }
            return exp[ind] / tot;
        }
    }
}
