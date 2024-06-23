using ImaginaryAI.StructuredAI;
using System.Text.Json;

namespace ImaginaryAI
{
    internal class Program
    {

        static void Main(string[] args)
        {
            int seed = 0;
            Random rand = new Random(seed);

            //Model testModel = new Model("Tests Model", [4, 3, 3, 2], softmax: true);
            double[][] biases = (double[][])JsonSerializer.Deserialize(
                File.ReadAllText("C:/Users/tommy/Desktop/biases.txt"),
                typeof(double[][]))!;
            double[][] weights = (double[][])JsonSerializer.Deserialize(
                File.ReadAllText("C:/Users/tommy/Desktop/weights.txt"),
                typeof(double[][]))!;
            Model testModel = new Model("Tests Model", biases, weights, cost: new CrossEntropyCost(),
                epsilon: 0.01, momentum: 0.0, hiddenLayerActivation: Activation.Sigmoid,
                outputLayerActivation: Activation.Softmax);
            //Model testModel = new Model("Tests Model", [784, 800, 10], cost: new CrossEntropyCost(),
            //    epsilon: 0.01, momentum: 0.0, hiddenLayerActivation: Activation.Sigmoid,
            //    outputLayerActivation: Activation.Softmax);
            //testModel.RandomizeModel(seed);

            Pass[] trainingData = ReadFormatted("C:/Users/tommy/Desktop/AI/Tests Model/mnist_train.csv");
            int rows = trainingData.Length;
            Pass[] testingData = ReadFormatted("C:/Users/tommy/Desktop/AI/Tests Model/mnist_test.csv");

            int batchSize = 10;
            int batches = (int)Math.Ceiling((double)rows / (double)batchSize);
            Trainer trainer = new Trainer(testModel, trainingData, epochs: 2000, batchSize: batchSize,
                (arr) =>
                {
                    // stretch/offset/rotate the image so that it is slightly different from the original.
                    if (OperatingSystem.IsWindows())
                        ImageProcessor.RandomScaleThatFits(arr, 28, rand, 0.9, 1.1
#if DEBUG
                            , writeImage: true
#endif
                );
                    else throw new ImaginaryException("The trainer only works on windows.");
                }
                );

            int ep = 0;
            int ba = 0;
            int curProgressBarChars = 0;
            trainer.Train(onBatchFinished: (correct) =>
            {
                ba++;
                double percentage = (double)ba / (double)batches;
                int filledChars = (int)Math.Floor(percentage * Extensions.ConsoleWidth);
                while (curProgressBarChars < filledChars)
                {
                    curProgressBarChars++;
                    '='.DumpSameLine();
                }
            }, onEpochFinished: () =>
            {
                ba = 0;
                curProgressBarChars = 0;
                double accuracy = trainer.Test(testingData, out double averageCost);
                ($"Epoch {ep} finished. " +
                $"Test accuracy: {accuracy:0.000}. " +
                $"Average cost: {averageCost:0.000}").Dump();
                ++ep;
                try
                {
                    testModel.ExportToDisk("C:/Users/tommy/Desktop/AI/");
                }
                catch (Exception e)
                {
                    e.Dump();
                }
            });
            //testModel.Dump();

            Console.ReadKey();
            Console.ReadKey();
            Console.ReadKey();
        }

        private static Pass[] ReadFormatted(string path)
        {
            string[] lines = File.ReadAllText(path).Split('\n')[1..^1];
            int rows = lines.Length;
            int[][] linesInt =
                (from l in lines
                 select (from one in l.Split(',', StringSplitOptions.RemoveEmptyEntries)
                         select int.Parse(one)).ToArray()).ToArray();

            return (from line in linesInt
                    select
                    new Pass(
                    inputs: (from one in line select (double)one / 255.0).ToArray()[1..],
                    expectedOutput: new double[10].SetAndReturn(line[0], 1.0))).ToArray();
        }
    }
}
