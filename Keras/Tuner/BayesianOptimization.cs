using Keras.Models;
using Python.Runtime;
using System;

namespace Keras.Tuner
{
    public class BayesianOptimization : BaseTuner
    {
        public BayesianOptimization(DefaultHyperModel hypermodel, string objective, int max_trials, int executions_per_trial = 1, int num_initial_points = 2, float alpha = 1e-4f, float beta = 2.6f, int? seed = null, HyperParameters hyperparameters = null, bool tune_new_entries = true, bool allow_new_entries = true, string directory = null, string project_name = null, LossFunctionWrapper loss = null, string[] metrics = null)
        {
            Parameters["hypermodel"] = hypermodel?.ToPython();
            Parameters["objective"] = objective;
            Parameters["max_trials"] = max_trials;
            Parameters["executions_per_trial"] = executions_per_trial;
            Parameters["num_initial_points"] = num_initial_points;
            Parameters["alpha"] = alpha;
            Parameters["beta"] = beta;
            Parameters["seed"] = seed;
            Parameters["hyperparameters"] = hyperparameters;
            Parameters["tune_new_entries"] = tune_new_entries;
            Parameters["allow_new_entries"] = allow_new_entries;
            Parameters["directory"] = directory;
            Parameters["metrics"] = metrics;
            Parameters["loss"] = loss;
            Parameters["project_name"] = project_name;

            PyInstance = Instance.kerastuner.tuners.bayesian.BayesianOptimization;
            Init();
        }


        public BayesianOptimization(Func<HyperParameters, BaseModel> hypermodelFunc, string objective, int max_trials, int executions_per_trial = 1, int num_initial_points = 2, float alpha = 1e-4f, float beta = 2.6f, int? seed = null, HyperParameters hyperparameters = null, bool tune_new_entries = true, bool allow_new_entries = true, string directory = null, string project_name = null, LossFunctionWrapper loss = null, string[] metrics = null)
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
            Parameters["executions_per_trial"] = executions_per_trial;
            Parameters["num_initial_points"] = num_initial_points;
            Parameters["alpha"] = alpha;
            Parameters["beta"] = beta;
            Parameters["seed"] = seed;
            Parameters["hyperparameters"] = hyperparameters;
            Parameters["tune_new_entries"] = tune_new_entries;
            Parameters["allow_new_entries"] = allow_new_entries;
            Parameters["directory"] = directory;
            Parameters["metrics"] = metrics;
            Parameters["loss"] = loss;
            Parameters["project_name"] = project_name;

            PyInstance = Instance.kerastuner.tuners.bayesian.BayesianOptimization;
            Init();
        }
    }
}
