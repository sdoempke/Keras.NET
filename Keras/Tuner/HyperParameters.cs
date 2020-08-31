using Python.Runtime;
using System.Collections.Generic;

namespace Keras.Tuner
{
    public class HyperParameters : Base
    {
        public HyperParameters()
        {
            PyInstance = Instance.kerastuner.engine.hyperparameters.HyperParameters();
        }

        public HyperParameters(PyObject pyObject)
        {
            PyInstance = pyObject;
        }

        public bool Boolean(string name, bool defaultValue=false, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["default"] = defaultValue;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Boolean", args);
            return py.As<bool>();
        }

        public int Int(string name, int min_value, int max_value, int step = 1, string sampling = null, int? defaultValue = null, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["min_value"] = min_value;
            args["max_value"] = max_value;
            args["step"] = step;
            args["sampling"] = sampling;
            args["default"] = defaultValue;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Int", args);
            return py.As<int>();
        }

        public double Float(string name, double min_value, double max_value, double step = 1, string sampling = null, double? defaultValue = null, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["min_value"] = min_value;
            args["max_value"] = max_value;
            args["step"] = step;
            args["sampling"] = sampling;
            args["default"] = defaultValue;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Float", args);
            return py.As<double>();
        }

        public bool Choice(string name, bool[] values, bool ordered = true, bool? defaultValue = null, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["values"] = values;
            args["ordered"] = ordered;
            args["default"] = defaultValue;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Choice", args);
            return py.As<bool>();
        }

        public int Choice(string name, int[] values, bool ordered = true, int? defaultValue = null, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["values"] = values;
            args["ordered"] = ordered;
            args["default"] = defaultValue;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Choice", args);
            return py.As<int>();
        }

        public double Choice(string name, double[] values, bool ordered = true, double? defaultValue = null, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["values"] = values;
            args["ordered"] = ordered;
            args["default"] = defaultValue;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Choice", args);
            return py.As<double>();
        }

        public string Choice(string name, string[] values, bool ordered = true, string defaultValue = null, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["values"] = values;
            args["ordered"] = ordered;
            args["default"] = defaultValue;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Choice", args);
            return py.As<string>();
        }

        public bool Fixed(string name, bool value, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["value"] = value;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Fixed", args);
            return py.As<bool>();
        }

        public int Fixed(string name, int value, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["value"] = value;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Fixed", args);
            return py.As<int>();
        }

        public double Fixed(string name, double value, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["value"] = value;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Fixed", args);
            return py.As<double>();
        }

        public string Fixed(string name, string value, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["value"] = value;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Fixed", args);
            return py.As<string>();
        }

        public PyObject Fixed(string name, PyObject value, string parent_name = null, string parent_values = null)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
            args["value"] = value;
            args["parent_name"] = parent_name;
            //args["parent_values"] = parent_values;

            PyObject py = InvokeMethod("Fixed", args);
            return py;
        }

        public T Get<T>(string name)
        {
            var args = new Dictionary<string, object>();
            args["name"] = name;
       
            PyObject py = InvokeMethod("get", args);
            return py.As<T>();
        }
    }
}
