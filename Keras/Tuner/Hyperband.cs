using Keras.Models;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Tuner
{
    public class Hyperband : BaseTuner
    {
        public Hyperband(DefaultHyperModel hypermodel, string objective, int max_epochs, int executions_per_trial = 1, int factor = 3, int hyperband_iterations = 1, int? seed = null, HyperParameters hyperparameters = null, bool tune_new_entries = true, bool allow_new_entries = true, string directory = null, string project_name = null, LossFunctionWrapper loss = null, string[] metrics = null)
        {
            Parameters["hypermodel"] = hypermodel?.ToPython();
            Parameters["objective"] = objective;
            Parameters["max_epochs"] = max_epochs;
            Parameters["executions_per_trial"] = executions_per_trial;
            Parameters["factor"] = factor;
            Parameters["hyperband_iterations"] = hyperband_iterations; 
            Parameters["seed"] = seed;
            Parameters["hyperparameters"] = hyperparameters;
            Parameters["tune_new_entries"] = tune_new_entries;
            Parameters["allow_new_entries"] = allow_new_entries;
            Parameters["directory"] = directory;
            Parameters["metrics"] = metrics;
            Parameters["loss"] = loss;
            Parameters["project_name"] = project_name;

            PyInstance = Instance.kerastuner.tuners.hyperband.Hyperband;
            Init();
        }


        public Hyperband(Func<HyperParameters, BaseModel> hypermodelFunc, string objective, int max_epochs, int executions_per_trial = 1, int factor = 3, int hyperband_iterations = 1, int? seed = null, HyperParameters hyperparameters = null, bool tune_new_entries = true, bool allow_new_entries = true, string directory = null, string project_name = null, LossFunctionWrapper loss = null, string[] metrics = null)
        {
            var func = new Func<PyObject, PyObject>((input) =>
            {
                using (Py.GIL())
                {
                    return hypermodelFunc?.Invoke(new HyperParameters(input))?.PyInstance;
                }
            });

            Parameters["hypermodel"] = func.ToPython();
            Parameters["objective"] = objective;
            Parameters["max_epochs"] = max_epochs;
            Parameters["executions_per_trial"] = executions_per_trial;
            Parameters["factor"] = factor;
            Parameters["hyperband_iterations"] = hyperband_iterations;
            Parameters["seed"] = seed;
            Parameters["hyperparameters"] = hyperparameters;
            Parameters["tune_new_entries"] = tune_new_entries;
            Parameters["allow_new_entries"] = allow_new_entries;
            Parameters["directory"] = directory;
            Parameters["metrics"] = metrics;
            Parameters["loss"] = loss;
            Parameters["project_name"] = project_name;

            PyInstance = Instance.kerastuner.tuners.hyperband.Hyperband;
            Init();
        }
    }
}
