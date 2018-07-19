using System;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;

namespace AvaloniaConcurrentDemo
{
	public class MainWindow : Window
	{
		ConcurrentLogger logger = new ConcurrentLogger();

		public MainWindow()
		{
			InitializeComponent();
#if DEBUG
			this.AttachDevTools();
#endif
			DoJob();
		}
		
		private void InitializeComponent()
		{
			AvaloniaXamlLoader.Load(this);
		}

		public void DoJob()
		{
			new Task(() =>
			{
				var tasks = Enumerable.Range(0, 3).Select(t => Task.Factory.StartNew(() =>
				{
					for (int i = 0; i < 1000; i++)
					{
						logger.PushMessage($"Message here! Thread {t}, value {i}");
					}
					logger.PushMessage($"----------Thread {t}: Job done!");
				})).ToArray();

				Task.WaitAll(tasks);
				logger.StopLogger();
				logger = null;
			}).Start();
		}
	}
}
