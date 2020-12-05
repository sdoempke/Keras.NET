using Keras.Datasets;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.Tuner;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Numpy;
using Python.Runtime;
using System;

namespace Keras.UnitTest
{
    /// <summary>
    /// Tests the Tuner with the cases given on the keras website
    /// https://keras-team.github.io/keras-tuner/examples/helloworld/
    /// </summary>
    [TestClass]
    public class KerasTunerHelloWorldTest
    {
        public readonly string Directory = @"c:\temp";


        #region Case 1

        [TestMethod]
        public void Case1()
        {
            var (x, y, val_x, val_y) = PrepareTestData();

            using (Py.GIL())
            {
                var tuner = new RandomSearch(BuildModel, "val_accuracy", 5, directory: Directory, project_name: "Case1");

                tuner.search_space_summary();

                tuner.Search(x, y, epochs: 3, validation_data: new NDarray[] { val_x, val_y });

                tuner.results_summary();
            }
        }

        #endregion

        #region Case 2

        [TestMethod]
        public void Case2()
        {
            var (x, y, val_x, val_y) = PrepareTestData();

            using (Py.GIL())
            {
                var tuner = new RandomSearch(BuildModel,
                    objective: "val_accuracy", 
                    loss: new SparseCategoricalCrossentropy(name:"my_loss"),
                    metrics: new[] { "accuracy", "mse" },
                    max_trials: 5, 
                    directory: Directory,
                    project_name: "Case2");

                tuner.search_space_summary();

                var stdResult = Keras.GetStdOut();
                Keras.ClearStdOut();

                tuner.Search(x, y, epochs: 5, validation_data: new NDarray[] { val_x, val_y });

                tuner.results_summary();

                stdResult = Keras.GetStdOut();
                Console.WriteLine(stdResult);
            }
        }

        #endregion

        #region Case 3

        [TestMethod]
        public void Case3()
        {
            var (x, y, val_x, val_y) = PrepareTestData();

            using (Py.GIL())
            {
                var myHyperModel = new MyHyperModel();

                var tuner = new RandomSearch(myHyperModel,
                    objective: "val_accuracy",
                    loss: new SparseCategoricalCrossentropy(name: "my_loss"),
                    metrics: new[] { "accuracy", "mse" },
                    max_trials: 5,
                    directory: Directory,
                    project_name: "Case3");

                tuner.search_space_summary();

                tuner.Search(x, y, epochs: 5, validation_data: new NDarray[] { val_x, val_y });

                tuner.results_summary();

                var stdResult = Keras.GetStdOut();
                Console.WriteLine(stdResult);
            }
        }

        #endregion

        #region Case 4

        [TestMethod]
        public void Case4()
        {
            var (x, y, val_x, val_y) = PrepareTestData();

            using (Py.GIL())
            {
                var hp = new HyperParameters();
                hp.Choice("learning_rate", new[] { 1e-1, 1e-3 });

                var tuner = new RandomSearch(BuildModel,
                    max_trials: 5,
                    hyperparameters: hp,
                    tune_new_entries: false,
                    directory: Directory,
                    project_name: "Case4",
                    objective: "val_accuracy");

                tuner.search_space_summary();

                tuner.Search(x, y, epochs: 5, validation_data: new NDarray[] { val_x, val_y });

                tuner.results_summary();

                var stdResult = Keras.GetStdOut();
                Console.WriteLine(stdResult);
            }
        }

        #endregion

        #region Case 5

        [TestMethod]
        public void Case5()
        {
            var (x, y, val_x, val_y) = PrepareTestData();

            using (Py.GIL())
            {
                var hp = new HyperParameters();
                hp.Fixed("learning_rate", 0.1f);

                var tuner = new RandomSearch(BuildModel,
                    max_trials: 5,
                    hyperparameters: hp,
                    tune_new_entries: true,
                    directory: Directory,
                    project_name: "Case5", 
                    objective: "val_accuracy");

                tuner.search_space_summary();

                tuner.Search(x, y, epochs: 5, validation_data: new [] { val_x, val_y });

                tuner.results_summary();

                var stdResult = Keras.GetStdOut();
                Console.WriteLine(stdResult);
            }
        }

        #endregion


        #region Case 6

        [TestMethod]
        public void Case6()
        {
            var (x, y, val_x, val_y) = PrepareTestData();

            using (Py.GIL())
            {
                var hp = new HyperParameters();
                hp.Choice("learning_rate", new[] { 1e-1, 1e-3 });

                var tuner = new RandomSearch(BuildModel,
                    max_trials: 5,
                    hyperparameters: hp,
                    tune_new_entries: true,
                    directory: Directory,
                    project_name: "Case6",
                    objective: "val_accuracy");

                tuner.search_space_summary();

                tuner.Search(x, y, epochs: 5, validation_data: new NDarray[] { val_x, val_y });

                tuner.results_summary();

                var stdResult = Keras.GetStdOut();
                Console.WriteLine(stdResult);
            }
        }

        #endregion

        #region Case 7

        [TestMethod]
        public void Case7()
        {
            var (x, y, val_x, val_y) = PrepareTestData();

            using (Py.GIL())
            {
                var hp = new HyperParameters();
                hp.Choice("learning_rate", new[] { 1e-1, 1e-3 });
                hp.Int("num_layers", 2, 20);

                var tuner = new RandomSearch(BuildModelCase7,
                    max_trials: 5,
                    hyperparameters: hp,
                    allow_new_entries: false,
                    directory: Directory,
                    project_name: "Case7",
                    objective: "val_accuracy");

                tuner.search_space_summary();

                tuner.Search(x, y, epochs: 5, validation_data: new NDarray[] { val_x, val_y });

                tuner.results_summary();

                var stdResult = Keras.GetStdOut();
                Console.WriteLine(stdResult);
            }
        }

        private BaseModel BuildModelCase7(HyperParameters hp)
        {
            var model = new Sequential();
            model.Add(new Flatten(input_shape: new Shape(28, 28)));
            for (int numLayers = 2; numLayers <= hp.Get<int>("num_layers"); numLayers++)
            {
                model.Add(new Dense(32, activation: "relu"));
            }

            model.Add(new Dense(10, activation: "softmax"));
            model.Compile(optimizer: new Adam(lr: hp.Get<float>("learning_rate")),
                          loss: "sparse_categorical_crossentropy",
                          metrics: new string[] { "accuracy" });

            return model;
        }

        #endregion

        #region Shared

        private BaseModel BuildModel(HyperParameters hp)
        {
            var model = new Sequential();
            model.Add(new Flatten(input_shape: new Shape(28, 28)));
            for (int numLayers = 2; numLayers <= hp.Int("num_layers", 2, 20); numLayers++)
            {
                model.Add(new Dense(hp.Int("units_" + numLayers, 32, 512, 32), activation: "relu"));
            }

            model.Add(new Dense(10, activation: "softmax"));
            model.Compile(optimizer: new Adam(lr: (float)hp.Choice("learning_rate", new[] { 1e-2, 1e-3, 1e-4 })),
                                        loss: "sparse_categorical_crossentropy",
                                        metrics: new string[] { "accuracy" });

            return model;
        }


        public class MyHyperModel : DefaultHyperModel
        {
            public override BaseModel Build(HyperParameters hp)
            {
                var model = new Sequential();
                model.Add(new Flatten(input_shape: new Shape(28, 28)));
                for (int numLayers = 2; numLayers <= hp.Int("num_layers", 2, 20); numLayers++)
                {
                    model.Add(new Dense(hp.Int("units_" + numLayers, 32, 512, 32), activation: "relu"));
                }

                model.Add(new Dense(10, activation: "softmax"));
                model.Compile(optimizer: new Adam(lr: (float)hp.Choice("learning_rate", new[] { 1e-2, 1e-3, 1e-4 })),
                              loss: "sparse_categorical_crossentropy", 
                              metrics: new string[] { "accuracy" });

                return model;
            }
        }

        public (NDarray, NDarray, NDarray, NDarray) PrepareTestData()
        {
            // the data, split between train and test sets
            var ((x, y), (val_x, val_y)) = MNIST.LoadData();

            x = x.astype(np.float32);
            val_x = val_x.astype(np.float32);
            x /= 255;
            val_x /= 255;

            x = x[":10000"];
            y = y[":10000"];

            return (x, y, val_x, val_y);
        }


        #endregion
    }
}
