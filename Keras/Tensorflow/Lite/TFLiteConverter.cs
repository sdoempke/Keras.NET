using Keras.Models;
using Python.Runtime;
using System;
using System.Collections.Generic;

namespace Keras.Tensorflow.Lite
{
    /// <summary>
    /// API Wrapper for Tensorflow lite converter
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class TFLiteConverter : Base
    {
        static dynamic caller = Instance.tensorflow.lite.TFLiteConverter;

        public static TFLiteConverter FromSavedModel(string savedModelFilePath)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();

            parameters.Add("saved_model_dir", savedModelFilePath);
            parameters.Add("signature_keys", null);
            parameters.Add("tags", null);

            return new TFLiteConverter((PyObject)InvokeStaticMethod(caller, "from_saved_model", parameters));
        }

        public static TFLiteConverter FromKerasModel(BaseModel model)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters.Add("model", model);
            return new TFLiteConverter((PyObject)InvokeStaticMethod(caller, "from_keras_model", parameters));
        }

        public TFLiteConverter(PyObject pyObject) : base()
        {
            PyInstance = pyObject;
        }

        public void SetDefaultOptimizations()
        {
            var pyArray = new PyObject[1];
            pyArray[0] = Keras.Instance.tensorflow.lite.Optimize.DEFAULT;
            var data = new PyList(pyArray);

            PyInstance.optimizations = data;
        }

        public void SetTargetSpecSupportedTypes(params Numpy.Dtype[] dtypes)
        {
            var pyArray = new PyObject[dtypes.Length];
            for(int index = 0; index < dtypes.Length; index++)
                pyArray[index] = dtypes[index].ToTensorflow();

            var data = new PyList(pyArray);
            PyInstance.target_spec.supported_types = data;
        }

        public void SetTargetSpecSupportedOpsToUInt8()
        {
            var pyArray = new PyObject[1];
            pyArray[0] = Keras.Instance.tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8;
            var data = new PyList(pyArray);
            PyInstance.target_spec.supported_ops = data;
        }

        public void SetInferenceInputType(Numpy.Dtype dtype)
        {
            PyInstance.inference_input_type = dtype.ToTensorflow();
        }

        public void SetInferenceOutputType(Numpy.Dtype dtype)
        {
            PyInstance.inference_output_type = dtype.ToTensorflow();
        }

        public void SetRepresentiveDataSet(float[] inputValues)
        {
            var func = new Func<PyObject>(() =>
            {
                using (Py.GIL())
                {
                    return ToList(inputValues);
                }
            });

            PyInstance.representative_dataset = func.ToPython();
        }

        public PyObject Convert()
        {
            return (PyObject)PyInstance.convert();
        }
    }
}
