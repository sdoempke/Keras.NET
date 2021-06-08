using Python.Runtime;
using System;

namespace Keras.Tensorflow.Distribute
{
    public class Strategy : Base
    {
        public StrategyScope Scope()
        {
            var scope = PyInstance.InvokeMethod("scope");
            return new StrategyScope(scope);
        }
    }

    public class StrategyScope : IDisposable
    {
        private PyObject _scope;

        internal StrategyScope(PyObject scope)
        {
            _scope = scope;
            Enter();
        }

        public void Dispose()
        {
            Exit();
        }

        private void Enter()
        {
            _scope.InvokeMethod("__enter__");
        }

        private void Exit()
        {
            // TypeError: __exit__() missing 3 required positional arguments: 'exception_type', 'exception_value', and 'traceback'
            _scope.InvokeMethod("__exit__", Runtime.None, Runtime.None, Runtime.None);
        }
    }
}
