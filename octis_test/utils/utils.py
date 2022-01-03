import numpy as np
import json

def get_best_hyperparameters(results_file):
    results = json.load(open(results_file, "r"))
    best_metric_value_idx = np.argmax(results["f_val"])
    metric = results["metric_name"]
    metric_key = f'best_{metric.lower()}_value'
    best_hyperparameters = {
        metric_key: results["f_val"][best_metric_value_idx],
    }
    for key, value in results["x_iters"].items():
        best_hyperparameters[key] = value[best_metric_value_idx]
    
    best_model_idx =  np.argmax(results["dict_model_runs"][metric][f'iteration_{best_metric_value_idx}'])
    best_hyperparameters["best_model_from_best_iteration"] = int(best_model_idx)
    best_hyperparameters["best_iteration"] = int(best_metric_value_idx)
    best_model_filename = f'{int(best_metric_value_idx)}_{int(best_model_idx)}.npz'
    best_hyperparameters["best_model_filename"] = best_model_filename
    results["best_hyperparameters"] = best_hyperparameters
    return results
