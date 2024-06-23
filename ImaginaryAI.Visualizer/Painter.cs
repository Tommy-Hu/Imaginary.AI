using Core.Graphing;
using Core.Graphing.GraphicObjects;
using Imaginary.Core;
using Imaginary.Core.ECS;
using Imaginary.Core.Util;
using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.Visualizer
{
    public class Painter : Component
    {
        private GraphicArea gArea;
        private GraphicPainting painting;

        public Painter(Entity parent, GraphicArea gArea, GraphicPainting painting) : base(parent)
        {
            this.gArea = gArea;
            this.painting = painting;
        }

        public override void Update()
        {
            base.Update();
            if (gArea.IsScreenPointInClient(Input.MousePos))
            {
                Vector2 localPoint = gArea.ScreenToLocalPoint(Input.MousePos);
                Vector2 pxOnPainting = localPoint * painting.Resolution.ToVector2() / gArea.AllowedSpaceInParent;
                int2 pxOnPaintingI = pxOnPainting.ToInt2();
                Color aaColor = Color.Lerp(Color.White, Color.Black, 0.5f);
                if (Input.IsMouseDown(0))
                {
                    painting.PaintPixel(pxOnPaintingI.X, pxOnPaintingI.Y, Color.White);
                    painting.PaintPixelIf(pxOnPaintingI.X - 1, pxOnPaintingI.Y, aaColor, (c) => c.R < aaColor.R);
                    painting.PaintPixelIf(pxOnPaintingI.X + 1, pxOnPaintingI.Y, aaColor, (c) => c.R < aaColor.R);
                    painting.PaintPixelIf(pxOnPaintingI.X, pxOnPaintingI.Y - 1, aaColor, (c) => c.R < aaColor.R);
                    painting.PaintPixelIf(pxOnPaintingI.X, pxOnPaintingI.Y + 1, aaColor, (c) => c.R < aaColor.R);
                }
                else if (Input.IsMouseDown(1))
                {
                    painting.PaintPixel(pxOnPaintingI.X, pxOnPaintingI.Y, Color.Black);
                    painting.PaintPixel(pxOnPaintingI.X - 1, pxOnPaintingI.Y, Color.Black);
                    painting.PaintPixel(pxOnPaintingI.X + 1, pxOnPaintingI.Y, Color.Black);
                    painting.PaintPixel(pxOnPaintingI.X, pxOnPaintingI.Y - 1, Color.Black);
                    painting.PaintPixel(pxOnPaintingI.X, pxOnPaintingI.Y + 1, Color.Black);
                }
            }
        }
    }
}
