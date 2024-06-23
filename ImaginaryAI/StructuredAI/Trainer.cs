using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    public class Trainer
    {
        public Model Model { get; }
        public Pass[] FormatedData { get; }
        public int Epochs { get; }
        public int BatchSize { get; }
        public int CurrentEpoch { get; private set; }
        public int CurrentBatch { get; private set; }
        public int CorrectsInCurrentEpoch { get; private set; }
        /// <summary>
        /// This function should modify (or keep) the double array such that it still represents the 
        /// same class as before modification. This allows variations in the training data and
        /// allow edge cases to be learned.
        /// </summary>
        public Action<double[]>? DataPreTrainingTransformer { get; private set; }

        /// <summary>
        /// Trains with the given csv.
        /// </summary>
        /// <param name="csvPath"></param>
        public Trainer(Model model, Pass[] formatedData, int epochs = 200, int batchSize = 10,
            Action<double[]>? dataPreTrainingTransformer = null)
        {
            Model = model;
            FormatedData = formatedData;
            Epochs = epochs;
            BatchSize = batchSize;
            DataPreTrainingTransformer = dataPreTrainingTransformer;
            //model.Dump(includeWeight: false);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Train(Action<int>? onBatchFinished = null, Action? onEpochFinished = null)
        {
            int rows = FormatedData.Length;
            int inputCount = Model.InputLayer.NeuronCount;

            for (int ep = 0; ep < Epochs; ep++)
            {
                Extensions.DumpHLine($"Epoch {ep}");

                int batchStart = 0;// the index of the first sample of the current batch
                while (batchStart < rows)
                {
                    Pass[] batch = new Pass[BatchSize];
                    int actualBatchSize = 0;
                    for (int b = 0; b < BatchSize; b++)
                    {
                        int sampleInd = batchStart + b;
                        if (sampleInd >= FormatedData.Length) break;
                        Debug.Assert(FormatedData[sampleInd].inputs.Length == inputCount);

                        actualBatchSize++;
                        batch[b] = FormatedData[sampleInd].Clone();
                        DataPreTrainingTransformer?.Invoke(batch[b].inputs);
                    }
                    if (actualBatchSize > 0)
                    {
                        int correct = Model.LearnBatch(batch);
                        onBatchFinished?.Invoke(correct);
                    }
                    batchStart += BatchSize;
                }

                onEpochFinished?.Invoke();
            }
        }

        /// <summary>
        /// Tests the model and returns the accuracy.
        /// </summary>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public double Test(Pass[] tests, out double averageCost)
        {
            int rows = tests.Length;
            int inputCount = Model.InputLayer.NeuronCount;

            int correctCount = 0;
            double totalCost = 0;
            for (int i = 0; i < rows; i++)
            {
                Pass pass = tests[i].Clone();
                DataPreTrainingTransformer?.Invoke(pass.inputs);
                Model.ForwardPass(pass);
                int predictedClass = Model.Prediction;
                if (predictedClass == pass.ExpectedClass)
                    correctCount++;
                totalCost += Model.GetResultCost(pass);
            }
            correctCount.Dump();
            averageCost = totalCost / (double)rows;
            return (double)correctCount / (double)rows;
        }

        /// <summary>
        /// This train function yields after every batch finishes.
        /// </summary>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public IEnumerator Train()
        {
            int rows = FormatedData.Length;
            int inputCount = Model.InputLayer.NeuronCount;

            for (int ep = 0; ep < Epochs; ep++)
            {
                int correct = 0;
                CurrentEpoch = ep;
                Extensions.DumpHLine($"Epoch {ep}");

                int batchStart = 0;// the index of the first sample of the current batch
                CurrentBatch = 0;
                while (batchStart < rows)
                {
                    Pass[] batch = new Pass[BatchSize];
                    int actualBatchSize = 0;
                    for (int b = 0; b < BatchSize; b++)
                    {
                        int sampleInd = batchStart + b;
                        if (sampleInd >= FormatedData.Length) break;
#if DEBUG
                        Debug.Assert(FormatedData[sampleInd].inputs.Length == inputCount);
#endif
                        actualBatchSize++;
                        batch[b] = FormatedData[sampleInd].Clone();
                        DataPreTrainingTransformer?.Invoke(batch[b].inputs);
                    }
                    if (actualBatchSize > 0)
                    {
                        correct += Model.LearnBatch(batch);
                        //yield return null;
                    }
                    batchStart += BatchSize;
                    CurrentBatch++;
                }
                CorrectsInCurrentEpoch = correct;
                yield return correct;
            }
        }
    }
}
