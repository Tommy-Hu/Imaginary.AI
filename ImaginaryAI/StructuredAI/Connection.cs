using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    /// <summary>
    /// Represents a connection between two neurons.
    /// </summary>
    public class Connection
    {
        public Neuron from;
        public Neuron to;

        public double weight;
        /// <summary>
        /// This value will be added to weight after all partials are calculated/accumulated.
        /// </summary>
        public double weightAdjustment;
        private double weightMomentum;


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public Connection(Neuron from, Neuron to)
        {
            this.from = from;
            this.to = to;
        }


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Wipe()
        {
            weight = 0;
            weightAdjustment = 0;
            weightMomentum = 0;
        }

        /// <summary>
        /// Calculates the value of this connection using the inputs from <see cref="from"/> and 
        /// using the bias from <see cref="to"/>.
        /// </summary>
        /// <returns></returns>

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public double CalculateConnectionValue()
        {
            return weight * from.activationValueCache;
        }


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void ApplyAdjustment(int batchSize, double momentum)
        {
            double changeThisPass = weightAdjustment / (double)batchSize + momentum * weightMomentum;
            weight += changeThisPass;
            weightMomentum = changeThisPass;
            weightAdjustment = 0;
        }
    }
}
