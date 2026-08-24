from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    non_zero = y_true != 0
    if non_zero.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)


def evaluate_forecasts(y_true, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    results = {}
    for model_name, y_pred in predictions.items():
        results[model_name] = {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE": mape(y_true, y_pred),
        }
    return results
