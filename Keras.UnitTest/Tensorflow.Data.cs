using Keras.Datasets;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.Tensorflow.Data;
using Keras.Tensorflow.Distribute;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Numpy;
using System;

namespace Keras.UnitTest
{
    [TestClass]
    public class TensorflowDatasetTest
    {
        [TestMethod]
        public void Test()
        {
            var (trainDataset, validationDataset) = PrepareTestData();
            Assert.IsNotNull(trainDataset);
            Assert.IsNotNull(validationDataset);

            BaseModel model = null;

            var strategy = new OneDeviceStrategy("CPU:0");

            using (var scoppe = strategy.Scope())
            {
                model = BuildTestModel();
            }

            trainDataset = trainDataset.Batch(64);
            validationDataset = validationDataset.Batch(64);
            var length = trainDataset.Length;

            var history = model.Fit(trainDataset, epochs: 30, validation_data: validationDataset);

            Console.WriteLine(history);
        }


        public BaseModel BuildTestModel()
        {

            var input = new Input(new Shape(28, 28));
            var x = new Flatten().Set(input);

            x = new Dense(512, activation: "relu").Set(x);

            var output = new Dense(10, activation: "softmax").Set(x);
            var model = new Model(new BaseLayer[] { input }, new BaseLayer[] { output });

            model.Compile(optimizer: new Adam(),
                          loss: "sparse_categorical_crossentropy",
                          metrics: new string[] { "accuracy" });

            return model;
        }

        public (Dataset, Dataset) PrepareTestData()
        {
            // the data, split between train and test sets
            var ((x, y), (val_x, val_y)) = MNIST.LoadData();

            x = x.astype(np.float32);
            val_x = val_x.astype(np.float32);
            x /= 255;
            val_x /= 255;

            x = x[":10000"];
            y = y[":10000"];

            return (Dataset.FromTensorSlices(x, y), Dataset.FromTensorSlices(val_x, val_y));
        }
    }
}
