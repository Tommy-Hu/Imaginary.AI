using Imaginary.Core.ECS;
using ImaginaryAI.StructuredAI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.Visualizer
{
    public class TrainerRunner : Component
    {
        public Trainer trainer;

        private Action onUpdate;

        public TrainerRunner(Entity parent, Trainer trainer, Action onUpdate) : base(parent)
        {
            this.trainer = trainer;
            this.onUpdate = onUpdate;
        }

        public void Begin()
        {
            StartCoroutine(trainer.Train());
        }

        public override void Update()
        {
            base.Update();
            onUpdate?.Invoke();
        }
    }
}
