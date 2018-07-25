using System;
using Avalonia;
using Avalonia.Logging.Serilog;
using AvaloniaMVVM.ViewModels;
using AvaloniaMVVM.Views;
using System.Runtime.InteropServices;

namespace AvaloniaMVVM
{
	class Program
	{
		static void Main(string[] args)
		{
			double x = Add(5.0, 17.0);
			Console.WriteLine($"5 + 17 = {x}");
			Console.WriteLine($"Current folder: {Environment.CurrentDirectory}");
			BuildAvaloniaApp().Start<MainWindow>(() => new MainWindowViewModel());
		}

		public static AppBuilder BuildAvaloniaApp()
			=> AppBuilder.Configure<App>()
				.UsePlatformDetect()
				.UseReactiveUI()
				.LogToDebug();

#if OS_LINUX
		[DllImport(@"./CMakeLibrary.so", EntryPoint = "Add")]
#endif
#if OS_WINDOWS
		[DllImport(@"./CMakeLibrary.dll", EntryPoint = "Add")]
#endif
		public static extern double Add(double x, double y);
	}
}
