using Keras.Models;
using Python.Runtime;
using System.Collections.Generic;
using System.Globalization;

namespace Keras
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

        public PyObject Convert()
        {
            return (PyObject)PyInstance.convert();
        }
    }
}
