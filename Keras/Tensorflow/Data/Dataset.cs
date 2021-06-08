using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Keras.Tensorflow.Data
{
    public class Dataset : Base
    {
        static dynamic caller = Instance.tensorflow.data.Dataset;

        public static Dataset FromTensorSlices(NDarray xTrain, NDarray yTrain = null)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();

            PyTuple tensors = null;
            if (yTrain != null)
                tensors = new PyTuple(new PyObject[] { xTrain.PyObject, yTrain.PyObject });
            else
                tensors = new PyTuple(new PyObject[] { xTrain.PyObject });

            parameters.Add("tensors", tensors);

            return new Dataset((PyObject)InvokeStaticMethod(caller, "from_tensor_slices", parameters));
        }

        public static Dataset FromTensorSlices(NDarray[] inputs, NDarray[] outputs = null)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();

            PyTuple tensors = null;
            var inputsTuple = new PyTuple(inputs.Select(p => (PyObject)p.PyObject).ToArray());
            if(outputs == null)
            {
                tensors = new PyTuple(new PyObject[] { inputsTuple });
            } 
            else
            {
                var outputsTuple = new PyTuple(outputs.Select(p => (PyObject)p.PyObject).ToArray());
                tensors = new PyTuple(new PyObject[] { inputsTuple, outputsTuple });
            }

            parameters.Add("tensors", tensors);

            return new Dataset((PyObject)InvokeStaticMethod(caller, "from_tensor_slices", parameters));
        }

        public static Dataset FromTensorSlices(Tuple<string, NDarray>[] inputs, Tuple<string, NDarray>[] outputs = null)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            PyTuple tensors = null;

            var pyInputsDict = new PyDict();
            foreach (var input in inputs)
                pyInputsDict[input.Item1] = input.Item2.PyObject;

            if(outputs == null)
            {
                tensors = new PyTuple(new PyObject[] { pyInputsDict });
            }
            else
            {
                var pyOutputsDict = new PyDict();
                foreach (var output in outputs)
                    pyOutputsDict[output.Item1] = output.Item2.PyObject;

                tensors = new PyTuple(new PyObject[] { pyInputsDict, pyOutputsDict });
            }

            parameters.Add("tensors", tensors);

            return new Dataset((PyObject)InvokeStaticMethod(caller, "from_tensor_slices", parameters));
        }

        public Dataset(PyObject pyObject) : base()
        {
            PyInstance = pyObject;
        }

        public Dataset Batch(int batchSize, bool drop_remainder = false)
        {
            var args = new Dictionary<string, object>();
            args["batch_size"] = batchSize;
            args["drop_remainder"] = drop_remainder;
        
            return new Dataset(InvokeMethod("batch", args));
        }

        public Dataset Repeat(int? count)
        {
            var args = new Dictionary<string, object>();
            args["count"] = count;
            return new Dataset(InvokeMethod("repeat", args));
        }

        public Dataset Shuffle(int buffer_size, bool? reshuffle_each_iteration = null)
        {
            var args = new Dictionary<string, object>();
           
            args["buffer_size"] = buffer_size;
            if (reshuffle_each_iteration.HasValue)
                args["reshuffle_each_iteration"] = reshuffle_each_iteration.Value;
            
            return new Dataset(InvokeMethod("shuffle", args));
        }

        public Dataset Cache(string filename= "")
        {
            var args = new Dictionary<string, object>();
            args["filename"] = filename;
            return new Dataset(InvokeMethod("cache", args));
        }

        public int Length
        {
            get
            {
                var result = PyInstance.InvokeMethod("__len__");
                return (int)result;
            }
        }
    }
}
