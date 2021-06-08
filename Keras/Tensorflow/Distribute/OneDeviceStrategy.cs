using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Tensorflow.Distribute
{
    public class OneDeviceStrategy : Strategy
    {
        public OneDeviceStrategy(string device) : base()
        {
            Parameters["device"] = device;

            PyInstance = Instance.tensorflow.distribute.OneDeviceStrategy;
            Init();
        }
    }
}
