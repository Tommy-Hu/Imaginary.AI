using Core.Graphing;
using Core.Graphing.GraphicObjects;
using Imaginary.Core;
using Imaginary.Core.ECS;
using Imaginary.Core.SceneManagement;
using Imaginary.Core.UI;
using Imaginary.Core.UI.Elements;
using Imaginary.Core.Util;
using ImaginaryAI.StructuredAI;
using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.Visualizer
{
    public class TwoDimensionalClassifier : Scene
    {
        public UILayer MainUI;

        public Model model;

        /// <summary>
        /// The maximum value the x coordinate of the input vector can be.
        /// </summary>
        const double SAMPLE_X_MAX = 1.0;
        /// <summary>
        /// The maximum value the y coordinate of the input vector can be.
        /// </summary>
        const double SAMPLE_Y_MAX = 1.0;

        public override void AfterLoaded()
        {
            MainUI = CreateUILayer("Main");
            model = new Model("2D Classifier", [2, 4, 3], cost: new CrossEntropyCost(), epsilon: .03,
                momentum: 0.0, hiddenLayerActivation: Activation.Sigmoid,
                outputLayerActivation: Activation.Softmax);
            int seed = 0;
            model.RandomizeModel(seed);
            Random random = new Random(seed);
            const int POINTS = 400;
            Pass[] formattedData = new Pass[POINTS];
            for (int i = 0; i < POINTS; i++)
            {
                var x = random.NextDouble() * SAMPLE_X_MAX;
                var y = random.NextDouble() * SAMPLE_Y_MAX;
                int pointClass = GetPointClass(x, y);
                double[] expected = new double[model.OutputLayer.NeuronCount];
                expected[pointClass] = 1.0;
                formattedData[i] = new Pass([x, y], expected);
            }
            Trainer trainer = new Trainer(model, formattedData, epochs: 2000, batchSize: 1);
            PlotPoint[] points = (from p in formattedData
                                  select new PlotPoint(p.inputs[0], p.inputs[1],
                                  GetPointClass(p.inputs[0], p.inputs[1]))).ToArray();

            PlotGraph plot = new PlotGraph(MainUI, MainUI.root, points,
                (0, SAMPLE_X_MAX), (0, SAMPLE_Y_MAX),
                [Color.Blue, Color.Red, Color.Green], pointThickness: 10,
                xAxisLabel: "Fruit Size (0.1 cm <-> 10 cm)",
                yAxisLabel: "Fruit Juiciness (dry as a rock <-> literal water ball)",
                title: "My AI and Plotting Library Showcase");

            plot.titleText.AlignmentX = AlignmentX.Center;

            VListElement informationPanelList = new VListElement(MainUI, plot.informationPanel);

            TextElement informationTitleElement = new TextElement(MainUI, informationPanelList);
            informationTitleElement.WeightInParent = 2;
            informationTitleElement.OverflowOptions = TextOverflowOptions.WrapAndIgnore;
            informationTitleElement.AlignmentX = AlignmentX.Left;

            TextElement informationContentElement = new TextElement(MainUI, informationPanelList);
            informationContentElement.WeightInParent = 3;
            informationContentElement.OverflowOptions = TextOverflowOptions.WrapAndIgnore;
            informationContentElement.Tint = Color.Gray;
            informationContentElement.AlignmentX = AlignmentX.Left;

            GraphicPainting painting = new GraphicPainting(
                (int)plot.gArea.AllowedSpaceInParent.X,
                (int)plot.gArea.AllowedSpaceInParent.Y);
            plot.gArea.AddGraphicBG(painting);

            Entity modelRunnerEntity = new Entity("Model Runner");
            EntityComponentManager.AddToRootEntity(modelRunnerEntity);
            TrainerRunner runner = new TrainerRunner(modelRunnerEntity, trainer, () =>
            {
                double totalCost = 0;
                int totalCorrect = 0;
                double PAINTING_AREA = (double)painting.Resolution.X * painting.Resolution.Y;
                for (int x = 0; x < painting.Resolution.X; x++)
                {
                    for (int y = 0; y < painting.Resolution.Y; y++)
                    {
                        double scaledX = ((double)x).Remap(0, painting.Resolution.X,
                            plot.xValueRange.Min, plot.xValueRange.Max);
                        double scaledY = ((double)y).Remap(0, painting.Resolution.Y,
                            plot.yValueRange.Min, plot.yValueRange.Max);
                        int pointClass = GetPointClass(scaledX, scaledY);
                        double[] expected = new double[model.OutputLayer.NeuronCount];
                        expected[pointClass] = 1.0;
                        var pass = new Pass([scaledX, scaledY], expected);
                        model.ForwardPass(pass);
                        int predictedClass = model.Prediction;
                        painting.PaintPixel(x, y, Color.Lerp(plot.classColors[predictedClass], Color.White, 0.7f));
                        totalCost += model.GetResultCost(pass);
                        if (predictedClass == pointClass)
                            totalCorrect++;
                    }
                }

                string informationTitle =
                $"Training my AI to predict if a Fruit is Lethal (red), Disgusting (blue), or Tasty (green)\n\n" +
                $"Training Accuracy: {trainer.CorrectsInCurrentEpoch * 100.0 /
                                      (double)formattedData.Length:0.000}%\n" +
                $"Testing Accuracy: {totalCorrect * 100.0 / PAINTING_AREA:0.000}%";
                informationTitleElement.Text = informationTitle;

                double[][] expBiases = model.ExportBiases();
                double[][] expWeights = model.ExportWeights();

                StringBuilder infoBuilder = new StringBuilder(
                $"Epochs: {trainer.CurrentEpoch}\n" +
                $"Batch: {trainer.CurrentBatch}\n" +
                $"Test Cost: {totalCost / PAINTING_AREA:0.00000}\n");
                infoBuilder.AppendLine("Biases, per layer");
                foreach (var layerBiases in expBiases)
                {
                    layerBiases.Dump(infoBuilder);
                    infoBuilder.AppendLine();
                }

                infoBuilder.AppendLine();
                infoBuilder.AppendLine("Weights, per connection-layer");
                foreach (var connectionLayerWeights in expWeights)
                {
                    connectionLayerWeights.Dump(infoBuilder);
                    infoBuilder.Append('\n');
                }

                informationContentElement.Text = $"{infoBuilder}";
            });

            runner.Begin();
        }

        private static int GetPointClass(double x, double y)
        {
            return x < 0.3 * SAMPLE_X_MAX || y > 0.8 ? x > 0.6 || y < 0.3 ? 0 : 1 : 2;
        }

        public override void BeforeLoaded()
        {
        }

        public override void OnUnloaded()
        {
        }

        #region Helpers
        private UILayer CreateUILayer(string layerName)
        {
            Entity layerEntity = new Entity("UILayer_" + layerName);
            EntityComponentManager.AddToRootEntity(layerEntity);
            EntityComponentManager.AddEntityAsPersistentUpdatable(layerEntity);
            return new UILayer(layerEntity);
        }
        #endregion
    }
}
