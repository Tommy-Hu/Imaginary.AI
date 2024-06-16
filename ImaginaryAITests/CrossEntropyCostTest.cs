using ImaginaryAI.StructuredAI;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit.Abstractions;

namespace ImaginaryAITests
{
    public class CrossEntropyCostTest
    {
        private readonly ITestOutputHelper output;

        public CrossEntropyCostTest(ITestOutputHelper output)
        {
            this.output = output;
        }

        [Fact]
        public void GetCost()
        {
            // test comes from https://machinelearningmastery.com/cross-entropy-for-machine-learning/

            var cost = new CrossEntropyCost();
            // define classification data
            double[] p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
            double[] q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3];
            // calculate cross entropy for each example
            List<double> results = new();
            (double, double, double)[] CORRECT =
                [
                    (1.0, 0.8, 0.223),
                    (1.0, 0.9, 0.105),
                    (1.0, 0.9, 0.105),
                    (1.0, 0.6, 0.511),
                    (1.0, 0.8, 0.223),
                    (0.0, 0.1, 0.105),
                    (0.0, 0.4, 0.511),
                    (0.0, 0.2, 0.223),
                    (0.0, 0.1, 0.105),
                    (0.0, 0.3, 0.357),

                ];
            for (int i = 0; i < p.Length; i++)
            {
                // create the distribution for each event {0, 1}
                double[] expected = [1.0 - p[i], p[i]];
                double[] predicted = [1.0 - q[i], q[i]];
                // calculate cross entropy for the two events
                double ce = cost.GetCost(new Pass(null!, expected), predicted);
                output.WriteLine($">[y={p[i]:0.0}, yhat={q[i]:0.0}] ce: {ce:0.000} nats");
                Assert.Equal(CORRECT[i].Item1, p[i], 1);
                Assert.Equal(CORRECT[i].Item2, q[i], 1);
                Assert.Equal(CORRECT[i].Item3, ce, 3);
                results.Add(ce);
            }

            // calculate the average cross entropy
            double mean_ce = results.Sum() / results.Count;
            output.WriteLine($"Average Cross Entropy: {mean_ce:0.000} nats");
            Assert.Equal(0.247, mean_ce, 3);
        }
    }
}
