using Keras.Models;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Tuner
{
    public class RandomSearch : BaseTuner
    {
        public RandomSearch(DefaultHyperModel hypermodel, string objective, int max_trials, int? seed = null, HyperParameters hyperparameters = null, bool tune_new_entries = true, bool allow_new_entries= true, string directory = null, string project_name = null, LossFunctionWrapper loss = null, string[] metrics = null)
        {
            Parameters["hypermodel"] = hypermodel?.ToPython();
            Parameters["objective"] = objective;
            Parameters["max_trials"] = max_trials;
            Parameters["hyperparameters"] = hyperparameters;
            Parameters["tune_new_entries"] = tune_new_entries;
            Parameters["allow_new_entries"] = allow_new_entries;
            Parameters["directory"] = directory;
            Parameters["metrics"] = metrics;
            Parameters["loss"] = loss;
            Parameters["project_name"] = project_name;

            PyInstance = Instance.kerastuner.tuners.RandomSearch;
            Init();
        }

        public RandomSearch(Func<HyperParameters, BaseModel> hypermodelFunc, string objective, int max_trials, int? seed = null, HyperParameters hyperparameters = null, bool tune_new_entries = true, bool allow_new_entries = true, string directory = null, string project_name = null, LossFunctionWrapper loss = null, string[] metrics = null) 
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
            Parameters["max_trials"] = max_trials;
            Parameters["hyperparameters"] = hyperparameters;
            Parameters["tune_new_entries"] = tune_new_entries;
            Parameters["allow_new_entries"] = allow_new_entries;
            Parameters["directory"] = directory;
            Parameters["metrics"] = metrics;
            Parameters["loss"] = loss;
            Parameters["project_name"] = project_name;

            PyInstance = Instance.kerastuner.tuners.RandomSearch;
            Init();
        }


    }
}
