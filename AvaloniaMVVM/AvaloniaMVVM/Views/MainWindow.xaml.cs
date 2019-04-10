using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
using AvaloniaMVVM.ViewModels;

namespace AvaloniaMVVM.Views
{
    public class MainWindow : Window
    {
        Image _img;

        public MainWindow()
        {
            InitializeComponent();
#if DEBUG
            this.AttachDevTools();
#endif

            // _img = this.FindControl<Image>("ImageCtrl");

            this.Closing += MainWindow_Closing;
        }

        private void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            ((MainWindowViewModel)DataContext).Dispose();
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);

            
        }
    }
}
