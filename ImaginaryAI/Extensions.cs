using ImaginaryAI.StructuredAI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI
{
    public static class Extensions
    {
        public static int ConsoleWidth
        {
            get
            {
                try
                {
                    return Console.WindowWidth;
                }
                catch
                {
                    return 30;
                }
            }
        }
        public static int ConsoleHeight
        {
            get
            {
                try
                {
                    return Console.WindowHeight;
                }
                catch
                {
                    return 30;
                }
            }
        }

        public static T[] SetAndReturn<T>(this T[] arr, int index, T value)
        {
            arr[index] = value;
            return arr;
        }

        public static void Dump<T>(this T[] arr)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append('[');
            for (int i = 0; i < arr.Length; i++)
            {
                sb.Append(arr[i]);
                if (i < arr.Length - 1) sb.Append(',').Append(' ');
            }
            sb.Append(']');
            Console.WriteLine(sb);
        }

        public static void Dump(this double[] arr, int decimalPlaces = 2)
        {
            StringBuilder sb = new StringBuilder();
            arr.Dump(sb, decimalPlaces);
            Console.WriteLine(sb);
        }

        public static void Dump(this double[] arr, StringBuilder sb, int decimalPlaces = 2)
        {
            sb.Append('[');
            for (int i = 0; i < arr.Length; i++)
            {
                sb.Append(arr[i].ToString("0." + new string('0', decimalPlaces)));
                if (i < arr.Length - 1) sb.Append(',').Append(' ');
            }
            sb.Append(']');
        }

        public static void Dump(this object obj)
        {
            Console.WriteLine(obj);
        }

        public static void DumpSameLine(this object obj)
        {
            Console.Write(obj);
        }

        public static void DumpHLine(string? title = null, char? fill = null)
        {
            Span<char> line = stackalloc char[ConsoleWidth];
            line.Fill(fill ?? '─');

            if (title != null)
            {
                const int START = 5;
                for (int i = START; i < line.Length && i - START < title.Length; i++)
                {
                    line[i] = title[i - START];
                }
            }
            Console.WriteLine(line.ToString());
        }

        public static void DumpSameLine(this double d, int decimalPlaces = 2)
        {
            Console.Write(d.ToString("0." + new string('#', decimalPlaces)));
        }

        public static void Dump(this Model model, bool includeWeight = true)
        {
            DumpHLine($"Model: {model.modelName}");
            for (int i = 0; i < model.layers.Length; i++)
            {
                Layer? layer = model.layers[i];
                DumpHLine($"Layer {i} Neurons", fill: '-');
                bool showBias = !layer.IsInputLayer && !layer.IsOutputLayer;
                bool showNeuronValue = !layer.IsInputLayer;
                bool showActivationValue = !layer.IsOutputLayer;
                List<string> items = [];
                if (showBias) items.Add("bias");
                if (showNeuronValue) items.Add("neuronValue");
                if (showActivationValue) items.Add("activationValue");

                $"Values here represent ({string.Join(',', items)})".Dump();
                layer.DumpSameLine();
                "".Dump();
                if (i > 0 && includeWeight)
                {
                    DumpHLine($"Layer {i - 1}->{i} Weights", fill: '-');
                    for (int j = 0; j < layer.NeuronCount; j++)
                    {
                        (from c in layer.neurons[j].fromConnections select c.weight).ToArray().Dump(decimalPlaces: 2);
                    }
                    "".Dump();
                }
                DumpHLine(fill: '-');
            }
            "Output: ".Dump();
            model.Results.Dump();
            DumpHLine(fill: '-');
            DumpHLine();
        }

        public static void DumpSameLine(this Layer layer)
        {
            foreach (var one in layer.neurons)
            {
                one.DumpSameLine();
                ' '.DumpSameLine();
            }
        }

        public static void DumpSameLine(this Neuron neuron)
        {
            '('.DumpSameLine();
            if (!neuron.layer.IsInputLayer)
            {
                if (!neuron.layer.IsOutputLayer)
                {
                    neuron.bias.DumpSameLine();
                    ','.DumpSameLine();
                }
                neuron.neuronValueCache.DumpSameLine();
                if (!neuron.layer.IsOutputLayer)
                    ','.DumpSameLine();
            }
            if (!neuron.layer.IsOutputLayer)
            {
                neuron.activationValueCache.DumpSameLine();
            }
            ')'.DumpSameLine();
        }
    }
}
