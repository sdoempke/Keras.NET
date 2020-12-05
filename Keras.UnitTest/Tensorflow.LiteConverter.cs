using System;
using System.Collections.Generic;
using System.Text;
using Python.Runtime;
using Keras;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Keras.Models;
using Keras.Layers;
using Keras.Optimizers;
using System.IO;

namespace Keras.UnitTest
{
    [TestClass]
    public class Tensorflow
    {
        [TestMethod]
        public void Test()
        {
            using (Py.GIL())
            {
                var model = BuildModel();
                var converter = TFLiteConverter.FromKerasModel(model);
                var tfLiteModel = converter.Convert();
                var bytes = tfLiteModel.As<byte[]>();
                File.WriteAllBytes(@"c:\temp\test.tflite", bytes);
                Console.WriteLine(""+tfLiteModel);
            }
        }

        public Sequential BuildModel()
        {
            var model = new Sequential();
            model.Add(new Flatten(input_shape: new Shape(28, 28)));
            model.Add(new Dense(512, activation: "relu"));
           
            model.Add(new Dense(10, activation: "softmax"));
            model.Compile(optimizer: new Adam(),
                                        loss: "sparse_categorical_crossentropy",
                                        metrics: new string[] { "accuracy" });

            return model;
        }
    }
}
