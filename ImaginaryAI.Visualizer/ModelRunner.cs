using Imaginary.Core.ECS;
using ImaginaryAI.StructuredAI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.Visualizer
{
    public class ModelRunner : Component
    {
        private Model model;
        /// <summary>
        /// The function should return a pass that will be executed through the model. Return null
        /// if no model should be ran.
        /// </summary>
        private Func<Pass?> onUpdate;
        /// <summary>
        /// The function will be invoked once the <see cref="onUpdate"/> pass is finished.
        /// </summary>
        private Action onPassFinished;

        public ModelRunner(Entity parent, Model model, Func<Pass?> onUpdate, Action onPassFinished) : base(parent)
        {
            this.model = model;
            this.onUpdate = onUpdate;
            this.onPassFinished = onPassFinished;
        }

        public override void Update()
        {
            base.Update();
            Pass? passNullable = onUpdate?.Invoke();
            if (passNullable is not null)
            {
                model.ForwardPass(passNullable.Value);
                onPassFinished?.Invoke();
            }
        }
    }
}
