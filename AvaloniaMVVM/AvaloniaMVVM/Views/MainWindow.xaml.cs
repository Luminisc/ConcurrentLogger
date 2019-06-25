using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using AvaloniaMVVM.ViewModels;

namespace AvaloniaMVVM.Views
{
    public class MainWindow : Window
    {
        private Image _img;
        private MainWindowViewModel Context => (MainWindowViewModel)DataContext;

        public MainWindow()
        {
            InitializeComponent();
#if DEBUG
            this.AttachDevTools();
#endif

            _img = this.FindControl<Image>("ImageCtrl");


            Closing += MainWindow_Closing;
            Opened += MainWindow_Opened;
        }

        private void MainWindow_Opened(object sender, System.EventArgs e)
        {
            _img.PointerPressed += (s, ev) =>
            {
                var pos = ev.GetPosition(_img);
                Context.OnImagePress(pos, new Size(_img.Bounds.Width, _img.Bounds.Height));
            };
        }

        private void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            Context.Dispose();
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);


        }
    }
}
