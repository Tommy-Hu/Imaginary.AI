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
            double[][] biases = (double[][])JsonSerializer.Deserialize(File.ReadAllText("C:/Users/tommy/Desktop/biases.txt"),
                typeof(double[][]))!;
            double[][] weights = (double[][])JsonSerializer.Deserialize(File.ReadAllText("C:/Users/tommy/Desktop/weights.txt"),
                typeof(double[][]))!;
            Model testModel = new Model("Tests Model", biases, weights, cost: new QuadraticCost(),
                epsilon: 0.01, momentum: 0.5);
            //Model testModel = new Model("Tests Model", [784, 800, 10], cost: new QuadraticCost(),
            //    epsilon: 0.01, momentum: 0.5);
            //testModel.RandomizeModel(seed, method: "Random");

            string[] lines = File.ReadAllText("C:/Users/tommy/Downloads/MNIST/mnist_train.csv").Split('\n')[1..^1];
            int rows = lines.Length;
            int[][] linesInt =
                (from l in lines
                 select (from one in l.Split(',', StringSplitOptions.RemoveEmptyEntries)
                         select int.Parse(one)).ToArray()).ToArray();

            Pass[] formatedData = (from line in linesInt
                                   select
                                   new Pass(
                                   inputs: (from one in line select (double)one / 255.0).ToArray()[1..],
                                   expectedOutput: new double[10].SetAndReturn(line[0], 1.0))).ToArray();


            int ep = 0;
            for (int i = 0; i < 10; i++)
            {
                Pass randomPass = formatedData[rand.Next(rows)];
                testModel.ForwardPass(randomPass);
                int predictedDigit = testModel.Prediction;
                int expectedDigit = randomPass.ExpectedClass;

                $"Epoch {ep} finished. Testing {expectedDigit}, got prediction {predictedDigit}.".Dump();
            }

            Trainer trainer = new Trainer(testModel, formatedData, epochs: 2000, batchSize: 1);

            trainer.Train(onEpochFinished: () =>
            {
                Pass randomPass = formatedData[rand.Next(rows)];
                testModel.ForwardPass(randomPass);
                int predictedDigit = testModel.Prediction;
                int expectedDigit = randomPass.ExpectedClass;

                $"Epoch {ep} finished. Testing {expectedDigit}, got prediction {predictedDigit}.".Dump();
            });
            //testModel.Dump();
            Console.ReadKey();
            Console.ReadKey();
            Console.ReadKey();
        }
    }
}
