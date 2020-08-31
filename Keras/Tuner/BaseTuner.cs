using Keras.Callbacks;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Tuner
{
    public class BaseTuner : Base
    {
        public void search_space_summary(bool extended = false)
        {
            var args = new Dictionary<string, object>();
            args["extended"] = extended;
          
            InvokeMethod("search_space_summary", args);
        }

        public void results_summary(int num_trials = 10)
        {
            var args = new Dictionary<string, object>();
            args["num_trials"] = num_trials;

            InvokeMethod("results_summary", args);
        }

        /// <summary>
        /// Trains the model for a given number of epochs (iterations on a dataset).
        /// </summary>
        /// <param name="x">Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs). If input layers in the model are named, you can also pass a dictionary mapping input names to Numpy arrays. x can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).</param>
        /// <param name="y">Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. y can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).</param>
        /// <param name="batch_size">Integer or None. Number of samples per gradient update. If unspecified, batch_sizewill default to 32.</param>
        /// <param name="epochs">Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.</param>
        /// <param name="verbose">Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.</param>
        /// <param name="callbacks">List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.</param>
        /// <param name="validation_split">Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.</param>
        /// <param name="validation_data">tuple (x_val, y_val) or tuple (x_val, y_val, val_sample_weights) on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split.</param>
        /// <param name="shuffle">Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.</param>
        /// <param name="class_weight">Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.</param>
        /// <param name="sample_weight">Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specifysample_weight_mode="temporal" in compile().</param>
        /// <param name="initial_epoch">Integer. Epoch at which to start training (useful for resuming a previous training run).</param>
        /// <param name="steps_per_epoch">Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.</param>
        /// <param name="validation_steps">Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.</param>
        /// <returns>A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).</returns>
        public void Search(NDarray x, NDarray y, int? batch_size = null, int epochs = 1, int verbose = 1, Callback[] callbacks = null,
                        float validation_split = 0.0f, NDarray[] validation_data = null, bool shuffle = true, Dictionary<int, float> class_weight = null,
                        NDarray sample_weight = null, int initial_epoch = 0, int? steps_per_epoch = null, int? validation_steps = null)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["batch_size"] = batch_size;
            args["epochs"] = epochs;
            args["verbose"] = verbose;
            args["callbacks"] = callbacks;
            args["validation_split"] = validation_split;
            if (validation_data != null)
            {
                if (validation_data.Length == 2)
                    args["validation_data"] = new PyTuple(new PyObject[] { validation_data[0].PyObject, validation_data[1].PyObject });
                else if (validation_data.Length == 3)
                    args["validation_data"] = new PyTuple(new PyObject[] { validation_data[0].PyObject, validation_data[1].PyObject, validation_data[2].PyObject });
            }

            args["shuffle"] = shuffle;
            if (class_weight != null)
                args["class_weight"] = ToDict(class_weight);
            args["sample_weight"] = sample_weight;
            args["initial_epoch"] = initial_epoch;
            args["steps_per_epoch"] = steps_per_epoch;
            args["validation_steps"] = validation_steps;

            PyObject py = InvokeMethod("search", args);
        }
    }
}
