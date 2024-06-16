using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    public class Model
    {
        public Layer[] layers;
        public string modelName;
        public double Epsilon { get; private set; }
        public Cost cost;

        public double Momentum { get; private set; }

        public Layer InputLayer => layers[0];
        public Layer OutputLayer => layers[^1];

        /// <summary>
        /// Constructs a new empty model.
        /// </summary>
        /// <param name="modelName"></param>
        /// <param name="neuronsPerLayer"></param>
        /// <param name="cost">The cost function of this model. If this is null, 
        /// <see cref="CrossEntropyCost"/> will be used.</param>
        /// <param name="epsilon">The learning rate. Larger is faster learning but less accurate.</param>
        /// <param name="softmax">Set this to true to use softmax before cost is applied.
        /// Note that this should only be used for catagorical models.</param>
        public Model(string modelName, int[] neuronsPerLayer, Cost? cost = null, double epsilon = 0.01,
            double momentum = 0.9)
        {
            this.modelName = modelName;
            this.Epsilon = epsilon;
            this.Momentum = momentum;
            this.cost = cost ?? new CrossEntropyCost();
            layers = new Layer[neuronsPerLayer.Length];
            for (int i = 0; i < neuronsPerLayer.Length; i++)
            {
                if (i > 0)
                {
                    layers[i] = new Layer(neuronsPerLayer[i], Activation.Sigmoid);
                    layers[i].Prev = layers[i - 1];
                }
                else
                {
                    // the input layer's activation values are used as the input sample vector.
                    layers[i] = new Layer(neuronsPerLayer[i], Activation.Error);
                }
            }
        }

        public Model(string modelName, double[][] neuronBiases, double[][] connectionWeights,
            Cost? cost = null, double epsilon = 0.01,
            double momentum = 0.9)
        {
            this.modelName = modelName;
            this.Epsilon = epsilon;
            this.Momentum = momentum;
            this.cost = cost ?? new CrossEntropyCost();

            layers = new Layer[neuronBiases.Length + 1];
            Debug.Assert(neuronBiases.Length == connectionWeights.Length);
            for (int i = 0; i < neuronBiases.Length + 1; i++)
            {
                if (i == 0) // input layer
                    layers[i] = new Layer(connectionWeights[0].Length / neuronBiases[0].Length, Activation.Error);
                else // hidden or output layer
                    layers[i] = new Layer(neuronBiases[i - 1].Length, Activation.Sigmoid);

                if (i > 0)
                    layers[i].Prev = layers[i - 1];
            }

            // set biases
            for (int i = 0; i < neuronBiases.Length; i++)
            {
                double[] layerBiases = neuronBiases[i];
                Layer layer = layers[i + 1];
                Debug.Assert(layerBiases.Length == layer.NeuronCount);
                for (int j = 0; j < layer.NeuronCount; j++)
                {
                    layer.neurons[j].Wipe();
                    layer.neurons[j].bias = layerBiases[j];
                }
            }

            // set weights
            for (int i = 0; i < connectionWeights.Length; i++)
            {
                int layerICount = layers[i].NeuronCount;
                Layer toLayer = layers[i + 1];
                for (int j = 0; j < toLayer.NeuronCount; j++)
                {
                    Neuron toNeuron = toLayer.neurons[j];
                    for (int k = 0; k < layerICount; k++)
                    {
                        Connection c = toNeuron.fromConnections![k];
                        c.Wipe();
                        c.weight = connectionWeights[i][k + j * layerICount];
                    }
                }
            }
        }

        /// <summary>
        /// Exports all the bias values in the model.
        /// Note that the input layer neurons are excluded because their biases are unused.
        /// </summary>
        /// <returns></returns>
        public double[][] ExportBiases()
        {
            double[][] res = new double[layers.Length - 1][];
            for (int i = 1; i < layers.Length; i++)
            {
                int layerNeurons = layers[i].NeuronCount;
                res[i - 1] = new double[layerNeurons];
                for (int j = 0; j < layerNeurons; j++)
                {
                    res[i - 1][j] = layers[i].neurons[j].bias;
                }
            }
            return res;
        }

        /// <summary>
        /// Exports all the weight values in the model.
        /// Returns an array of layer-connections.
        /// Each item in the array is an array of weights in that layer-connection, and they are
        /// ordered by (prev_0-cur_0, prev_1-cur_0, prev_2-cur_0, ..., prev_n-cur_0, prev_0-cur_1, prev_1-cur_1, ...).
        /// </summary>
        /// <returns></returns>
        public double[][] ExportWeights()
        {
            double[][] res = new double[layers.Length - 1][];
            for (int i = 1; i < layers.Length; i++)
            {
                int layerNeurons = layers[i].NeuronCount;
                int prevLayerNeurons = layers[i - 1].NeuronCount;
                double[] prevToCurLayerWeights = new double[layers[i - 1].NeuronCount * layerNeurons];
                for (int j = 0; j < layerNeurons; j++)
                {
                    Neuron cur = layers[i].neurons[j];
                    Connection[] connToJthCurNeuron = cur.fromConnections!;
                    for (int k = 0; k < connToJthCurNeuron.Length; k++)
                    {
                        prevToCurLayerWeights[j * prevLayerNeurons + k] = connToJthCurNeuron[k].weight;
                    }
                }
                res[i - 1] = prevToCurLayerWeights;
            }
            return res;
        }

        /// <summary>
        /// Completely randomizes (i.e., resets) the entire model.
        /// </summary>
        public void RandomizeModel(int seed, string? method = "Random")
        {
            Random random = new Random(seed);
            if (method == "Xavier")
            {
                // Xavier Weight Initialization
                for (int i = 1; i < layers.Length; i++)
                {
                    int n = layers[i - 1].NeuronCount;
                    var (lower, upper) = (-(1.0 / Math.Sqrt(n)), (1.0 / Math.Sqrt(n)));
                    RandomizeLayer(random, upper, lower, i);
                }
            }
            else if (method == "Normalized Xavier")
            {
                // Normalized Xavier Weight Initialization
                for (int i = 1; i < layers.Length; i++)
                {
                    int n = layers[i - 1].NeuronCount;
                    int m = layers[i].NeuronCount;
                    var (lower, upper) = (-(Math.Sqrt(6) / Math.Sqrt(n + m)), (Math.Sqrt(6) / Math.Sqrt(n + m)));
                    RandomizeLayer(random, upper, lower, i);
                }
            }
            else
            {
                // Random Initialization
                for (int i = 1; i < layers.Length; i++)
                {
                    var (lower, upper) = (-1.0, 1.0);
                    //var (lower, upper) = (Math.Pow(10, -9), Math.Pow(10, -8));
                    RandomizeLayer(random, upper, lower, i);
                }

            }
        }

        private void RandomizeLayer(Random random, double upper, double lower, int layerInd)
        {
            if (layerInd == 0) throw new ImaginaryException("You may not randomize the input layer!");
            int n = layers[layerInd - 1].NeuronCount;
            int m = layers[layerInd].NeuronCount;
            for (int j = 0; j < m; j++)
            {
                //if (layerInd != layers.Length - 1)
                //{
                //    // randomize bias
                //    double r = 0;
                //    while (r == 0)
                //    {
                //        r = random.NextDouble() * (upper - lower) + lower;
                //    }
                //    layers[layerInd].neurons[j].bias = r;
                //    layers[layerInd].neurons[j].biasAdjustment = 0;
                //}
                for (int k = 0; k < n; k++)
                {
                    double r = 0;
                    while (r == 0)
                    {
                        r = random.NextDouble() * (upper - lower) + lower;
                    }

                    layers[layerInd].neurons[j].fromConnections![k].weight = r / Math.Sqrt(n);
                    layers[layerInd].neurons[j].fromConnections![k].weightAdjustment = 0;
                }
            }
        }

        public void ForwardPass(Pass pass)
        {
            for (int i = 0; i < InputLayer.NeuronCount; i++)
            {
                Neuron? neuron = InputLayer.neurons[i];
                neuron.activationValueCache = pass.inputs[i];
            }

            InputLayer.ForwardStep();
        }

        /// <summary>
        /// Backwards and propagates (learning is done here).
        /// </summary>
        /// <param name="pass"></param>
        public void BackwardPass(Pass pass)
        {
            double[] predictedOutput = Results;
            double[] initialDerivatives = new double[OutputLayer.NeuronCount];

            for (int i = 0; i < OutputLayer.NeuronCount; i++)
            {
                initialDerivatives[i] = cost.GetCostDerivative(pass, predictedOutput, i) * -Epsilon;
            }

            OutputLayer.BackwardStep(initialDerivatives);
        }

        /// <summary>
        /// Learns the given batch of passes and returns the number of correctly classified passes.
        /// </summary>
        /// <param name="batch"></param>
        /// <returns></returns>
        public int LearnBatch(Pass[] batch)
        {
            int batchSize = 0;
            int correct = 0;
            for (int i = 0; i < batch.Length; i++)
            {
                if (batch[i].inputs is null || batch[i].expectedOutput is null)
                    break;// incomplete batch. break.
                batchSize++;
                ForwardPass(batch[i]);
                if (batch[i].ExpectedClass == Prediction)
                    correct++;
                BackwardPass(batch[i]);
            }

            // Apply learning.
            foreach (var layer in layers)
            {
                foreach (var n in layer.neurons)
                {
                    n.ApplyAdjustment(batchSize, this.Momentum);
                    if (!layer.IsInputLayer)
                    {
                        foreach (var c in n.fromConnections!)
                        {
                            c.ApplyAdjustment(batchSize, this.Momentum);
                        }
                    }
                }
            }
            return correct;
        }

        public double[] OutputNeuronValues => (from one in OutputLayer.neurons select one.neuronValueCache).ToArray();
        public double[] Results => (from one in OutputLayer.neurons select one.activationValueCache).ToArray();

        public int Prediction
        {
            get
            {
                double[] results = Results;
                double maxSoFar = double.NegativeInfinity;
                int maxInd = 0;
                for (int i = 0; i < results.Length; i++)
                {
                    if (results[i] > maxSoFar)
                    {
                        maxSoFar = results[i];
                        maxInd = i;
                    }
                }
                return maxInd;
            }
        }

        public double GetResultCost(Pass pass) => cost.GetCost(pass, Results);
    }
}
