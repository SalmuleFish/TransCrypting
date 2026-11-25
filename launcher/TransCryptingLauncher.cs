using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

namespace TransCryptingLauncher
{
    internal static class Program
    {
        [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = true)]
        private static extern int SetCurrentProcessExplicitAppUserModelID(string appID);

        private static string QuoteArg(string arg)
        {
            if (string.IsNullOrEmpty(arg))
            {
                return "\"\"";
            }
            if (arg.IndexOfAny(new[] { ' ', '\t', '"' }) == -1)
            {
                return arg;
            }
            var sb = new StringBuilder();
            sb.Append('"');
            int backslashes = 0;
            foreach (char ch in arg)
            {
                if (ch == '\\')
                {
                    backslashes++;
                }
                else if (ch == '"')
                {
                    sb.Append('\\', backslashes * 2 + 1);
                    sb.Append('"');
                    backslashes = 0;
                }
                else
                {
                    if (backslashes > 0)
                    {
                        sb.Append('\\', backslashes);
                        backslashes = 0;
                    }
                    sb.Append(ch);
                }
            }
            if (backslashes > 0)
            {
                sb.Append('\\', backslashes * 2);
            }
            sb.Append('"');
            return sb.ToString();
        }

        [STAThread]
        private static int Main(string[] args)
        {
            try
            {
                SetCurrentProcessExplicitAppUserModelID("TransCrypting.LocalApp");
            }
            catch
            {
            }

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string pythonwExe = Path.Combine(baseDir, ".venv", "Scripts", "pythonw.exe");
            string pythonCliExe = Path.Combine(baseDir, ".venv", "Scripts", "python.exe");
            string pythonExe = File.Exists(pythonwExe) ? pythonwExe : pythonCliExe;
            string runPy = Path.Combine(baseDir, "run.py");

            if (!File.Exists(pythonExe))
            {
                MessageBox.Show(
                    "No se encontró el entorno virtual en:\n" + pythonExe +
                    "\n\nEjecuta primero:\n  python -m venv .venv\n  .venv\\Scripts\\pip install -r requirements.txt",
                    "TransCrypting — Falta .venv",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }

            if (!File.Exists(runPy))
            {
                MessageBox.Show(
                    "No se encontró el archivo principal:\n" + runPy,
                    "TransCrypting — Archivo faltante",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }

            var argParts = new List<string> { QuoteArg(runPy) };
            foreach (string a in args)
            {
                argParts.Add(QuoteArg(a));
            }

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = string.Join(" ", argParts),
                WorkingDirectory = baseDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

            var stderrTail = new Queue<string>();
            var stderrLock = new object();

            try
            {
                using (var proc = new Process { StartInfo = psi })
                {
                    proc.OutputDataReceived += (sender, e) => { };
                    proc.ErrorDataReceived += (sender, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            lock (stderrLock)
                            {
                                stderrTail.Enqueue(e.Data);
                                while (stderrTail.Count > 15)
                                {
                                    stderrTail.Dequeue();
                                }
                            }
                        }
                    };

                    if (!proc.Start())
                    {
                        MessageBox.Show(
                            "No se pudo iniciar Python para ejecutar TransCrypting.",
                            "TransCrypting — Error de inicio",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Error
                        );
                        return 1;
                    }

                    proc.BeginOutputReadLine();
                    proc.BeginErrorReadLine();
                    proc.WaitForExit();

                    if (proc.ExitCode != 0)
                    {
                        string details;
                        lock (stderrLock)
                        {
                            details = string.Join("\n", stderrTail);
                        }
                        if (!string.IsNullOrWhiteSpace(details))
                        {
                            MessageBox.Show(
                                "TransCrypting finalizó con un error (código " + proc.ExitCode + "):\n\n" + details,
                                "TransCrypting — Error",
                                MessageBoxButtons.OK,
                                MessageBoxIcon.Error
                            );
                        }
                    }
                    return proc.ExitCode;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Error al iniciar TransCrypting:\n" + ex.Message,
                    "TransCrypting — Error",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }
        }
    }
}
