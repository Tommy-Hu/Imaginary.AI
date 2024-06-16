using ImaginaryAI.StructuredAI;

namespace ImaginaryAITests
{
    public class SoftmaxTests
    {
        private double[] mtArr => [];
        private double[] zeroArr => [0];
        private double[] oneArr => [1];
        private double[] minusArr => [-1];
        private double[] simpleArr => [1, 2, 3, 4, 5];
        private double[] minusSimpleArr => [-1, -2, -3, -4, -5];
        private double[] centeredArr => [-3, -2, -1, 0, 1, 2, 3];

        [Fact]
        public void Calculate()
        {
            Assert.Empty(Softmax.Calculate(mtArr));
            Assert.Collection(Softmax.Calculate(zeroArr), [(d) => Assert.Equal(1.0, d, 5)]);
            Assert.Collection(Softmax.Calculate(oneArr), [(d) => Assert.Equal(1.0, d, 5)]);
            Assert.Collection(Softmax.Calculate(minusArr), [(d) => Assert.Equal(1.0, d, 5)]);

            Assert.Equal(1.0, Softmax.Calculate(simpleArr).Sum(), 5);
            Assert.Equal(1.0, Softmax.Calculate(minusSimpleArr).Sum(), 5);
            Assert.Equal(1.0, Softmax.Calculate(centeredArr).Sum(), 5);
        }
    }
}