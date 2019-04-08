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

            _img = this.FindControl<Image>("ImageCtrl");
            this.Opened += MainWindow_Opened;
        }

        private void MainWindow_Opened(object sender, System.EventArgs e)
        {
            _img.Source = ((MainWindowViewModel)DataContext).RenderImage;
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);

            
        }
    }
}
