using Imaginary.Core;
using Imaginary.Core.Audio;
using Imaginary.Core.IO;
using Imaginary.Core.Rendering;
using Imaginary.Core.SceneManagement;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using static System.Net.Mime.MediaTypeNames;
using System;
using Imaginary.Core.UI;
using Imaginary.Core.UI.Elements;

namespace ImaginaryAI.Visualizer
{
    public class Visualizer : Game
    {

        public static Visualizer Current;

        private readonly GraphicsDeviceManager graphics;
        private SpriteBatch spriteBatch;

        public Visualizer()
        {
            PathsHolder.GameName = "AI";
            Writer.CreateDefaultWriteFolder();

            Current = this;
            graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
            IsMouseVisible = false;
        }

        protected override void Initialize()
        {
            //graphics.SynchronizeWithVerticalRetrace = false;
            IsFixedTimeStep = false;
            TargetElapsedTime = TimeSpan.FromSeconds(1d / 60d);

            //GraphicsAdapter.UseReferenceDevice = true;

            graphics.PreferredBackBufferWidth = 1300;
            graphics.PreferredBackBufferHeight = 900;
            Window.Title = "Imaginary AI - Visualizer";


            graphics.ApplyChanges();
            Window.AllowUserResizing = true;
            Window.ClientSizeChanged += OnResize!;
            base.Initialize();//this calls LoadContent();
        }

        protected override void LoadContent()
        {
            spriteBatch = new SpriteBatch(GraphicsDevice);
            Gizmos.ReInit(graphics);

#if RELEASE || PLAYTEST || DEMO
            const bool ASYNC_INITIALIZATION = true;
#else
            const bool ASYNC_INITIALIZATION = false;
#endif
            //GameCore.Initializer = Initializer.Init;
            GameCore.Initialize(spriteBatch, Content, graphics, GraphicsDevice, Window, this, () =>
            {
                Cursor.Init();
                SceneManager.Init();//initializes sceneManager

                SFXManager.Init();// init sound effects
                MusicManager.Init();// init music manager

                //SceneManager.QueueLoadScene(new TwoDimensionalClassifier());
                SceneManager.QueueLoadScene(new MNIST());

                Input.Init(this);

                GameCore.ShowSystemCursor = true;
                GameCore.PlayInBackground = true;
            }, ASYNC_INITIALIZATION);//load resources needed and load default resources
        }

        private void OnResize(object sender, EventArgs e)
        {
            GameCore.OnResizedWindow(graphics);
        }

        protected override void Update(GameTime updateTime)
        {
            base.Update(updateTime);

            GameCore.UpdateFrame(updateTime);
        }

        protected override void Draw(GameTime drawTime)
        {
            base.Draw(drawTime);

            GameCore.DrawFrame(drawTime);
        }

        protected override void OnExiting(object sender, EventArgs args)
        {
            foreach (var target in GraphicsDevice.GetRenderTargets())
            {
                target.RenderTarget.Dispose();
            }
            GameCore.StopGame();
            base.OnExiting(sender, args);
        }
    }
}
