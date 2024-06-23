using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    public class Neuron
    {
        public Connection[]? fromConnections;
        public Connection[]? toConnections;
        public Layer layer;

        public double bias;
        /// <summary>
        /// This value will be added to bias after all partials are calculated/accumulated.
        /// </summary>
        public double biasAdjustment;
        private double biasMomentum;

        public double activationValueCache;
        public double neuronValueCache;

        /// <summary>
        /// Clears all data in this neuron.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Wipe()
        {
            bias = 0;
            biasAdjustment = 0;
            biasMomentum = 0;
            activationValueCache = 0;
            neuronValueCache = 0;
        }

        /// <summary>
        /// This function is also known as the z() function.
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="inputs"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void RecalculateNeuronValue()
        {
            // note that the input layer neurons are skipped from the Step function below
            neuronValueCache = 0;
            if (!layer.IsInputLayer)
            {
                foreach (var connection in fromConnections!)
                {
                    double z = connection.CalculateConnectionValue();
                    neuronValueCache += z;
                }
                neuronValueCache += bias;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void SetActivationValue(double newValue)
        {
            activationValueCache = newValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void ApplyAdjustment(int batchSize, double momentum)
        {
            double changeThisPass = biasAdjustment / (double)batchSize + momentum * biasMomentum;
            bias += changeThisPass;
            biasMomentum = changeThisPass;
            biasAdjustment = 0;
        }
    }
}
