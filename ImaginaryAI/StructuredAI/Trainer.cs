using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
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
        /// Trains with the given csv.
        /// </summary>
        /// <param name="csvPath"></param>
        public Trainer(Model model, Pass[] formatedData, int epochs = 200, int batchSize = 10)
        {
            Model = model;
            FormatedData = formatedData;
            Epochs = epochs;
            BatchSize = batchSize;

            //model.Dump(includeWeight: false);
        }

        public void Train(Action? onBatchFinished = null, Action? onEpochFinished = null)
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
                        batch[b] = FormatedData[sampleInd];
                    }
                    if (actualBatchSize > 0)
                    {
                        Model.LearnBatch(batch);
                        onBatchFinished?.Invoke();
                    }
                    batchStart += BatchSize;
                }

                onEpochFinished?.Invoke();
            }
        }

        /// <summary>
        /// This train function yields after every batch finishes.
        /// </summary>
        /// <returns></returns>
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
                        batch[b] = FormatedData[sampleInd];
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
