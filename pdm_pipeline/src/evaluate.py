"""
evaluate.py
Standard metrics for the classification (health-state) and regression
(RUL) tasks, matching Chapter 3.8 of the project report.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
)


def classification_report_dict(y_true, y_pred, labels=("Critical", "Warning", "Healthy")):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "labels_order": list(labels),
    }


def regression_report_dict(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse_hours": rmse,
        "mae_hours": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def metrics_table(results: dict) -> pd.DataFrame:
    """results: {model_name: metrics_dict} -> a tidy comparison dataframe."""
    return pd.DataFrame(results).T
