using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    public class Layer
    {
        public Neuron[] neurons;

        public int NeuronCount => neurons.Length;

        private Layer? prev;
        private Layer? next;

        public Layer? Prev
        {
            get => prev;
            set
            {
                if (value != null)
                {
                    prev = value;
                    prev.Next = this;
                }
                else prev = null;
            }
        }

        public Layer? Next
        {
            get => next;
            set
            {
                next = value;
                if (next != null)
                {
                    next.prev = this;
                    for (int j = 0; j < next.NeuronCount; j++)
                        next.neurons[j].fromConnections = new Connection[NeuronCount];
                    for (int i = 0; i < NeuronCount; i++)
                    {
                        neurons[i].toConnections = new Connection[next.NeuronCount];
                        for (int j = 0; j < next.NeuronCount; j++)
                        {
                            var connection = new Connection(neurons[i], next.neurons[j]);
                            neurons[i].toConnections![j] = connection;
                            next.neurons[j].fromConnections![i] = connection;
                        }
                    }
                }
            }
        }

        public bool IsInputLayer => prev == null;
        public bool IsOutputLayer => next == null;

        public Layer(int neuronCount, Activation activation)
        {
            neurons = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                neurons[i] = new Neuron(activation) { layer = this, };
            }
        }

        public void ForwardStep()
        {
            foreach (var n in neurons)
            {
                n.Step();
            }
            if (!IsOutputLayer)
                next!.ForwardStep();
        }

        public void BackwardStep(double[] partials)
        {
            Debug.Assert(partials.Length == NeuronCount);

            if (IsInputLayer) return;

            // Note that partials are calculated until (and excluding) the neuron activation function derivative.
            // Now, each of these values must be multiplied by its corresponding da/dz z_curLayer.

            for (int i = 0; i < NeuronCount; i++)
            {
                // satisfy the vec(da/dz (z_curLayer)) invariant, where z_curLayer are
                // z values of this layer.
                Neuron n = neurons[i];
                partials[i] *= n.activation.activationDerivative(n.neuronValueCache);
            }

            // Now, partials contains all partial derivative up to (and including) a'(z_i).

            double[] aAdjust = new double[prev.NeuronCount];
            for (int i = 0; i < NeuronCount; i++)
            {
                Neuron n = neurons[i];
                // INVARIANT:
                // partial is -Epsilon * dCost/dK_i (output_i) *
                //      (LOOP n times: {vec(W_nextLayerConn) DOT vec(da/dz (z_curLayer))}),
                // where n is the number of layers passed, and when n is zero, the second part of the equation
                // is not evaluated (i.e., second part is equal to 1). Here, DOT is the dot product.
                double partial = partials[i];
                for (int j = 0; j < prev.NeuronCount; j++)
                {
                    Connection c = n.fromConnections![j];

                    // Adjust a
                    // This is also done to satisfy the invariant vec(W_nextLayerConn), where W_nextLayerConn
                    // are connection weights between this layer and the previous layer because this layer
                    // becomes the next layer in the next step.
                    // Here, aAdjust[j] is the j'th item in W_nextLayerConn (i.e., c.weight in code). Therefore,
                    aAdjust[j] += partial * c.weight;

                    // Adjust W
                    // Equation on iPad is: 
                    // W += partials_j * a, where a is the previous layer's j'th neuron's activation value.
                    n.fromConnections![j].weightAdjustment += partial * c.from.activationValueCache;

                    // Adjust b
                    n.biasAdjustment += partial;
                }
            }

            // Here aAdjust (is a vector that) is W_nextLayerConn, where W_nextLayerConn are weights that connects
            // this layer and the previous layer.
            prev.BackwardStep(aAdjust);
        }
    }
}
