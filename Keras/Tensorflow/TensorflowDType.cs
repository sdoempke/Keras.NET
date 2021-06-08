using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Tensorflow
{
    public static class NumpyDTypEntensions 
    {
        public static Python.Runtime.PyObject ToTensorflow(this Numpy.Dtype dType)
        {
            var dTypeString = dType.ToString();

            if(dTypeString.Equals(Numpy.np.float16.ToString()))
                 return Keras.Instance.tensorflow.float16;
            if (dTypeString.Equals( Numpy.np.float32.ToString()))
                return Keras.Instance.tensorflow.float32;
            if (dTypeString.Equals(Numpy.np.uint8.ToString()))
                return Keras.Instance.tensorflow.uint8;
            if (dTypeString.Equals(Numpy.np.int8.ToString()))
                return Keras.Instance.tensorflow.int8;

            return Keras.Instance.tensorflow.float32;
        }
    }
}
