using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    [Serializable]
    public class ImaginaryException : Exception
    {
        public ImaginaryException() { }
        public ImaginaryException(string message) : base(message) { }
        public ImaginaryException(string message, Exception inner) : base(message, inner) { }
        protected ImaginaryException(
          System.Runtime.Serialization.SerializationInfo info,
          System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
    }
}
