import numpy as np
import json

def get_best_hyperparameters(results_file):
    results = json.load(open(results_file, "r"))
    best_metric_value_idx = np.argmax(results["f_val"])
    metric_key = f'best_{results["metric_name"].lower()}_value'
    best_hyperparameters = {
        metric_key: results["f_val"][best_metric_value_idx],
    }
    for key, value in results["x_iters"].items():
        best_hyperparameters[key] = value[best_metric_value_idx]
    
    results["best_hyperparameters"] = best_hyperparameters
    return results
