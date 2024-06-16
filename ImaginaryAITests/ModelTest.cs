using ImaginaryAI.StructuredAI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAITests
{
    public class ModelTest
    {
        [Fact]
        public void Constructor()
        {
            Model model = new Model("TEST", [], [], cost: null, epsilon: 0.01, softmax: true);
        }
    }
}
