using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    public class Activation
    {
        /// <summary>
        /// Represents a function that takes an array of neuron values and converts it into an array
        /// of activation values.
        /// </summary>
        /// <param name="neuronValues"></param>
        /// <returns>An array of activation values.</returns>
        public delegate double[] ActivationFunction(double[] neuronValues);
        /// <summary>
        /// Represents a function that takes an array of neuron values and an array of previous partial
        /// derivatives and returns an array of new partial derivatives that chains previous 
        /// <paramref name="partials"/> with the partial derivatives of the activation function.
        /// </summary>
        /// <param name="neuronValues"></param>
        /// <param name="partials"></param>
        /// <returns>An array of new partial derivatives that is the result of the chain of
        /// <paramref name="partials"/> and the partial derivative of this activation function.</returns>
        public delegate double[] ActivationDerivative(double[] neuronValues, double[] partials);

        /// <summary>
        /// Converts an array of neuron values into an array of activation values.
        /// </summary>
        public readonly ActivationFunction activationFunc;
        /// <summary>
        /// Converts an array of neuron values and an array of partials into an array of activation derivatives.
        /// </summary>
        public readonly ActivationDerivative activationDerivative;


        private static Activation? sigmoid = null;
        public static Activation Sigmoid => sigmoid ??= new Activation(A_Sigmoid, AP_SigmoidDerivative);
        private static Activation? error = null;
        public static Activation Error => error ??= new Activation(_Error, _Error);
        private static Activation? softmax = null;
        public static Activation? Softmax => softmax ??= new Activation(A_Softmax, AP_SoftmaxDerivative);


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private Activation(ActivationFunction activationFunc, ActivationDerivative activationDerivative)
        {
            this.activationFunc = activationFunc;
            this.activationDerivative = activationDerivative;
        }


        [MethodImpl(MethodImplOptions.AggressiveOptimization)] private static double[] _Error(double[] neuronValues) => throw new ImaginaryException("Error!");

        [MethodImpl(MethodImplOptions.AggressiveOptimization)] private static double[] _Error(double[] neuronValues, double[] partials) => throw new ImaginaryException("Error!");


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static double[] A_Sigmoid(double[] neuronValues)
        {
            // here, I use the sigmoid / logistic function, but can also use other functions like tanh
            double[] results = new double[neuronValues.Length];
            for (int i = 0; i < neuronValues.Length; i++)
            {
                results[i] = 1.0 / (1.0 + Math.Exp(-neuronValues[i]));
            }
            return results;
        }


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static double[] AP_SigmoidDerivative(double[] neuronValues, double[] partials)
        {
            // the derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
            double[] newPartials = new double[neuronValues.Length];
            double[] sx = A_Sigmoid(neuronValues);
            for (int i = 0; i < neuronValues.Length; i++)
            {
                newPartials[i] = sx[i] * (1.0 - sx[i])
                                    * partials[i] /*Apply chain rule*/;
            }
            return newPartials;
        }


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static double[] A_Softmax(double[] neuronValues)
        {
            // neuronValues is an array that represents the output vector.
            if (neuronValues.Length == 0) return null!;// something went wrong

            double tot = 0.0;
            double[] exp = new double[neuronValues.Length];
            double max = neuronValues.Max();
            for (int i = 0; i < neuronValues.Length; i++)
            {
                exp[i] = Math.Exp(neuronValues[i] - max);
                tot += exp[i];
            }
            for (int i = 0; i < exp.Length; i++)
            {
                exp[i] /= tot;
            }

            return exp;
        }

        /// <summary>
        /// Helper function.
        /// </summary>
        /// <param name="neuronValues"></param>
        /// <returns></returns>

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static double[,] SoftmaxDerivative(double[] neuronValues)
        {
            double[] softmax = A_Softmax(neuronValues);
            int n = softmax.Length;
            double[,] jacobian = new double[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j)
                    {
                        jacobian[i, j] = softmax[i] * (1 - softmax[i]);
                    }
                    else
                    {
                        jacobian[i, j] = -softmax[i] * softmax[j];
                    }
                }
            }
            return jacobian;
        }


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static double[] AP_SoftmaxDerivative(double[] neuronValues, double[] partials)
        {
            double[,] jacobian = SoftmaxDerivative(neuronValues);
            return CombineGradients(partials, jacobian);
        }


        /// <summary>
        /// Helper function.
        /// </summary>

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static double[] CombineGradients(double[] partials, double[,] jacobian)
        {
            int n = partials.Length;
            double[] newPartials = new double[n];

            for (int i = 0; i < n; i++)
            {
                newPartials[i] = 0;
                for (int j = 0; j < n; j++)
                {
                    newPartials[i] += jacobian[i, j] * partials[j];
                }
            }
            return newPartials;
        }
    }
}
