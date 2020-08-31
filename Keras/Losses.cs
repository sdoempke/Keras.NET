using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    /// <summary>
    /// A loss function (or objective function, or optimization score function) is one of the two parameters required to compile a model
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Losses : Base
    {
        static dynamic caller = Instance.keras.losses;

        /// <summary>
        /// Calculates the mean squared error.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray MeanSquaredError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;

            return new NDarray(InvokeStaticMethod(caller, "mean_squared_error", parameters));
        }

        /// <summary>
        /// Calculates the mean absolute error.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray MeanAbsoluteError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;

            return new NDarray(InvokeStaticMethod(caller, "mean_absolute_error", parameters));
        }

        /// <summary>
        /// Calculates the mean absolute percentage error.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray MeanAbsolutePercentageError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "mean_absolute_percentage_error", parameters));
        }

        /// <summary>
        /// Calculates the mean squared log error.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray MeanSquaredLogarithmicError(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "mean_squared_logarithmic_error", parameters));
        }

        /// <summary>
        /// Calculates the Square Hinge
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray SquaredHinge(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "squared_hinge", parameters));
        }

        /// <summary>
        /// Calculates the Hinge error.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray Hinge(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "hinge", parameters));
        }

        /// <summary>
        /// Calculates the categorial hinge.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray CategoricalHinge(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "categorical_hinge", parameters));
        }

        /// <summary>
        /// Logarithm of the hyperbolic cosine of the prediction error.
        /// log(cosh(x)) is approximately equal to(x** 2) / 2 for small x and to abs(x) - log(2) for large x.This means that 'logcosh' works mostly like the mean squared error, but will not be so strongly affected by the occasional wildly incorrect prediction.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray LogCosh(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "logcosh", parameters));
        }

        /// <summary>
        /// Categoricals the crossentropy.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray CategoricalCrossentropy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "categorical_crossentropy", parameters));
        }

        /// <summary>
        /// Sparses the categorical crossentropy.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray SparseCategoricalCrossentropy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "sparse_categorical_crossentropy", parameters));
        }

        /// <summary>
        /// Binaries the crossentropy.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray BinaryCrossentropy(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "binary_crossentropy", parameters));
        }

        /// <summary>
        /// Kullbacks the leibler divergence.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray KullbackLeiblerDivergence(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "kullback_leibler_divergence", parameters));
        }

        /// <summary>
        /// Poissons the specified y true.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray Poisson(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "poisson", parameters));
        }

        /// <summary>
        /// Cosines the proximity.
        /// </summary>
        /// <param name="y_true">tensor of true targets.</param>
        /// <param name="y_pred">tensor of predicted targets.</param>
        /// <returns></returns>
        public static NDarray CosineProximity(NDarray y_true, NDarray y_pred)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["y_true"] = y_true;
            parameters["y_pred"] = y_pred;
            return new NDarray(InvokeStaticMethod(caller, "cosine_proximity", parameters));
        }
    }


    /// <summary>
    /// LossFunctionWrapper
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public abstract class LossFunctionWrapper : Base
    {
        /// <summary>
        /// Invokes the `LossFunctionWrapper` instance.
        /// </summary>
        /// <param name="y_true">Ground truth values.</param>
        /// <param name="y_pred">The predicted values.</param>
        /// <returns>Loss values per sample</returns>
        NDarray Call(NDarray y_true, NDarray y_pred)
        {
            var args = new Dictionary<string, object>();
            args["y_true"] = y_true;
            args["y_pred"] = y_pred;
        
            var pyResult = InvokeMethod("call", args);
            return pyResult.As<NDarray>();
        }
    }

    /// <summary>
    /// Computes the crossentropy loss between the labels and predictions.
    /// Use this crossentropy loss function when there are two or more label classes.
    /// We expect labels to be provided as integers.If you want to provide labels
    /// using `one-hot` representation, please use `CategoricalCrossentropy` loss.
    /// There should be `# classes` floating point values per feature for `y_pred`
    /// and a single floating point value per feature for `y_true`.
    /// </summary>
    /// <seealso cref="Keras.LossFunctionWrapper" />
    public class SparseCategoricalCrossentropy : LossFunctionWrapper
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SparseCategoricalCrossentropy"/> class.
        /// </summary>
        /// <param name="from_logits">Whether `y_pred` is expected to be a logits tensor. By default, we assume that `y_pred` encodes a probability distribution. **Note - Using from_logits = True may be more numerically stable.</param>
        /// <param name="reduction">(Optional) Type of `tf.keras.losses.Reduction` to apply to loss.Default value is `AUTO`. `AUTO` indicates that the reduction option will be determined by the usage context.For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When used with `tf.distribute.Strategy`, outside of built-in training loops such as `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error.Please see this custom training [tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training) for more details.</param>
        /// <param name="name">: Optional name for the op. Defaults to 'sparse_categorical_crossentropy'</param>
        public SparseCategoricalCrossentropy(bool from_logits = false,
                                             string reduction = "auto",
                                             string name = "sparse_categorical_crossentropy")
        {
            Parameters["from_logits"] = from_logits;
            Parameters["reduction"] = reduction;
            Parameters["name"] = name;

            PyInstance = Instance.keras.losses.SparseCategoricalCrossentropy;
            Init();
        }
    }
}
