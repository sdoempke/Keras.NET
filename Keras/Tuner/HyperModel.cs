using Keras.Models;
using Python.Runtime;
using System;

namespace Keras.Tuner
{
    public abstract class DefaultHyperModel : Base
    {
        private Func<PyObject, PyObject> _buildAction;
     
        public DefaultHyperModel(string name = null, bool tunable = true)
        {
            _buildAction = new Func<PyObject, PyObject>(InternalBuild);
            Parameters["build"] =  _buildAction.ToPython();
            Parameters["name"] = name;
            Parameters["tunable"] = tunable;

            PyInstance = Keras.Instance.kerastuner.engine.hypermodel.DefaultHyperModel;
            Init();
        }

        private PyObject InternalBuild(PyObject hp)
        {
            using (Py.GIL())
            {
                return Build(new HyperParameters(hp))?.PyInstance;
            }
        }

        public abstract BaseModel Build(HyperParameters hyperParameters);
    }
}
