using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace AvaloniaConcurrentDemo
{
    class ConcurrentLogger
    {
	    private bool _loggerEnabled = false;
	    readonly ConcurrentQueue<string> _logQueue = new ConcurrentQueue<string>();
	    private Task _loggingTask;

		public ConcurrentLogger()
	    {
		    SetupLogger();
		}

	    public void PushMessage(string text)
	    {
			_logQueue.Enqueue(text);
	    }

	    private void SetupLogger()
	    {
		    _loggerEnabled = true;
		    _loggingTask = new Task(() =>
		    {
			    using (Stream stream = new FileStream("./output.log", FileMode.OpenOrCreate))
			    using (TextWriter tw = new StreamWriter(stream))
			    {
				    while (_loggerEnabled)
				    {
					    if (_logQueue.TryDequeue(out var text))
					    {
						    tw.WriteLine(text);
					    }
				    }
			    }
		    });
		    _loggingTask.Start();
	    }
	}
}
