using Core.Graphing;
using Core.Graphing.GraphicObjects;
using Imaginary.Core;
using Imaginary.Core.ECS;
using Imaginary.Core.SceneManagement;
using Imaginary.Core.UI;
using Imaginary.Core.UI.Data;
using Imaginary.Core.UI.Elements;
using Imaginary.Core.Util;
using ImaginaryAI.StructuredAI;
using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace ImaginaryAI.Visualizer
{
    public class MNIST : Scene
    {
        public UILayer MainUI;

        public override void AfterLoaded()
        {
            MainUI = CreateUILayer("Main");
            MainUI.IgnoreAllInputsInLayer = false;
            MainUI.root.BackColor = ColorLib.Gray;

            double[][] biases = (double[][])JsonSerializer.Deserialize(
                File.ReadAllText("biases.txt"),
                typeof(double[][]))!;
            double[][] weights = (double[][])JsonSerializer.Deserialize(
                File.ReadAllText("weights.txt"),
                typeof(double[][]))!;
            Model model = new Model("MNIST Sig-Quad Model", biases, weights, cost: new CrossEntropyCost(),
                epsilon: 0.01, momentum: 0.0, hiddenLayerActivation: Activation.Sigmoid,
                outputLayerActivation: Activation.Softmax);

            VListElement vlist = new VListElement(MainUI, MainUI.root);
            vlist.IgnoreMouseInput = true;

            TextElement titleElement = new TextElement(MainUI, vlist);

            AspectBoxElement gAreaAbox = new AspectBoxElement(MainUI, vlist);
            gAreaAbox.IgnoreAllInputIncludingChildren = true;
            gAreaAbox.Ratio = 1;
            gAreaAbox.FitType = AspectBoxElement.FitMode.Best;
            gAreaAbox.WeightInParent = 5;
            MarginBoxElement gAreaMbox = new MarginBoxElement(MainUI, gAreaAbox);
            gAreaMbox.MarginIsPercentage = true;
            gAreaMbox.MarginLT = gAreaMbox.MarginRB = Vector2.One * 0.05f;
            gAreaMbox.BackColor = ColorLib.Gray2Less;
            GraphicArea gArea = new GraphicArea(MainUI, gAreaMbox);

            MarginBoxElement clearButtonABox = new MarginBoxElement(MainUI, vlist);
            clearButtonABox.MarginIsPercentage = true;
            clearButtonABox.MarginLT = clearButtonABox.MarginRB = new Vector2(0.4f, 0.2f);
            SpriteTextButtonElement clearButton = new SpriteTextButtonElement(MainUI, clearButtonABox);
            clearButton.Texts = new InteractableTexts("Clear");
            clearButton.BgColors = InteractableColors.WhiteAndYellow;
            clearButton.textElement.FontSize = 40;

            TextElement predictionTitleElement = new TextElement(MainUI, vlist);
            TextElement predictionTextElement = new TextElement(MainUI, vlist);
            predictionTextElement.WeightInParent = 2;

            titleElement.Text = "Handwriting Digit Recognition";
            predictionTitleElement.Text = "I think it's a ";
            predictionTextElement.Text = "0";
            predictionTitleElement.OverflowOptions = TextOverflowOptions.Fit;
            predictionTextElement.OverflowOptions = TextOverflowOptions.Fit;

            var painting = new GraphicPainting(28, 28, true);
            painting.Fill(Color.Black);
            gArea.AddGraphicBG(painting);

            clearButton.OnMouseClick += (btn, e) =>
            {
                painting.Fill(Color.Black);
            };

            Entity painterEntity = new Entity("Painter");
            EntityComponentManager.AddToRootEntity(painterEntity);
            Painter painter = new Painter(painterEntity, gArea, painting);

            Entity modelRunnerEntity = new Entity("Runner");
            EntityComponentManager.AddToRootEntity(modelRunnerEntity);
            ModelRunner runner = new ModelRunner(modelRunnerEntity, model, () =>
            {
                Color[] data = painting.GetScaledPaintingData(28, 28);
                double[] buffer = (from d in data select (double)d.R / 255.0).ToArray();
                if (OperatingSystem.IsWindows())
                {
                    var btm = ImageProcessor.BufferToBitmap(buffer, 28, 28);
                    btm.Save("C:/Users/tommy/Desktop/AI/Tests Model/Painter.png",
                        System.Drawing.Imaging.ImageFormat.Png);
                    btm.Dispose();
                }
                return new Pass(buffer, new double[10]);
            },
            () =>
            {
                int digit = model.Prediction;
                predictionTextElement.Text = digit.ToString();
            });
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
