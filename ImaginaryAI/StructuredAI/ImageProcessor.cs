using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;

namespace ImaginaryAI.StructuredAI
{
    [SupportedOSPlatform("windows")]
    public class ImageProcessor
    {
        /// <summary>
        /// Resize the image to the specified width and height.
        /// </summary>
        /// <param name="image">The image to resize.</param>
        /// <param name="width">The width to resize to.</param>
        /// <param name="height">The height to resize to.</param>
        /// <returns>The resized image.</returns>

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void RandomScaleThatFits(double[] input, int width, Random rng,
            double maxDownscaleAmount = 0.5, double maxUpscaleAmount = 2.0
#if DEBUG
            , bool writeImage = true
#endif
            )
        {
            int n = input.Length;
            int height = n / width;

            int leftMost = width - 1;
            int rightMost = 0;
            int topMost = height - 1;
            int bottomMost = 0;

            GetBoundingBox(input, width, ref leftMost, ref rightMost, ref topMost, ref bottomMost);
            // we found the corners of the image.
            // randomly select a scaling option that fits.
            int minWidth = Math.Max(1, (int)(width * maxDownscaleAmount));
            int maxWidth = Math.Min((int)(width * maxUpscaleAmount), (int)(width - (rightMost - leftMost + 1) + width));
            int minHeight = Math.Max(1, (int)(height * maxDownscaleAmount));
            int maxHeight = Math.Min((int)(height * maxUpscaleAmount), (int)(height - (bottomMost - topMost + 1) + height));
            // put the pixel data into the image.
            Bitmap image = BufferToBitmap(input, width, height);
#if DEBUG
            if (writeImage)
                image.Save("C:/Users/tommy/Desktop/AI/Tests Model/OriginalImg.png");
#endif

            Bitmap resized = ResizeImage(image, rng.Next(minWidth, maxWidth + 1), rng.Next(minHeight, maxHeight + 1));
            image.Dispose();
            GetBoundingBox(resized, ref leftMost, ref rightMost, ref topMost, ref bottomMost);
            int newBoundWidth = rightMost - leftMost + 1;
            int newBoundHeight = bottomMost - topMost + 1;

            newBoundWidth = Math.Min(newBoundWidth, width);
            newBoundHeight = Math.Min(newBoundHeight, height);

            int newTopLeftX = rng.Next(width - newBoundWidth + 1);
            int newTopLeftY = rng.Next(height - newBoundHeight + 1);

            // copy the resized data.
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (x >= newTopLeftX && x < newTopLeftX + newBoundWidth &&
                        y >= newTopLeftY && y < newTopLeftY + newBoundHeight)
                        input[y * width + x] =
                            (double)resized.GetPixel(
                                x - newTopLeftX + leftMost,
                                y - newTopLeftY + topMost)
                            .R / 255.0;
                    else
                        input[y * width + x] = 0.0;
                }
            }
            resized.Dispose();

#if DEBUG
            if (writeImage)
            {
                Bitmap img = BufferToBitmap(input, width, height);
                img.Save("C:/Users/tommy/Desktop/AI/Tests Model/ProcessedImg.png");
                img.Dispose();
            }
#endif
        }


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static Bitmap BufferToBitmap(double[] input, int width, int height)
        {
            Bitmap image = new Bitmap(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int brightness = (int)(255 * input[y * width + x]);
                    image.SetPixel(x, y, Color.FromArgb(255, brightness, brightness, brightness));
                }
            }

            return image;
        }


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void GetBoundingBox(double[] unedited, int width,
            ref int leftMost, ref int rightMost, ref int topMost, ref int bottomMost)
        {
            int height = unedited.Length / width;

            leftMost = width - 1;
            rightMost = 0;
            topMost = height - 1;
            bottomMost = 0;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (unedited[y * width + x] > 0.0)
                    {
                        leftMost = Math.Min(leftMost, x);
                        rightMost = Math.Max(rightMost, x);
                        topMost = Math.Min(topMost, y);
                        bottomMost = Math.Max(bottomMost, y);
                    }
                }
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void GetBoundingBox(Bitmap unedited,
            ref int leftMost, ref int rightMost, ref int topMost, ref int bottomMost)
        {
            int height = unedited.Height;
            int width = unedited.Width;

            leftMost = width - 1;
            rightMost = 0;
            topMost = height - 1;
            bottomMost = 0;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (unedited.GetPixel(x, y).R > 0)
                    {
                        leftMost = Math.Min(leftMost, x);
                        rightMost = Math.Max(rightMost, x);
                        topMost = Math.Min(topMost, y);
                        bottomMost = Math.Max(bottomMost, y);
                    }
                }
            }
        }
    }
}
