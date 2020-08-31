using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.UnitTest
{
    [TestClass]
    public class BootstrapTestEnvironment
    {
        public const string PythonPath = @"C:\DeepLearning\Python38";

        [AssemblyInitialize]
        public static void TestInitialize(TestContext testContext)
        {
            Setup.UseTfKeras();
            Keras.DisablePySysConsoleLog = true;

            string path = PythonPath + ";" + Environment.GetEnvironmentVariable("PATH", EnvironmentVariableTarget.Machine);
            Environment.SetEnvironmentVariable("PATH", path, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONHOME", PythonPath, EnvironmentVariableTarget.Process);
            Environment.SetEnvironmentVariable("PYTHONPATH ", PythonPath, EnvironmentVariableTarget.Process);
        }
    }
}
