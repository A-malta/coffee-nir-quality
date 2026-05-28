import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from typing import Union


def _specificity_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    specificities = []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - (tp + fn + fp)

        denom = tn + fp
        if denom > 0:
            specificities.append(tn / denom)

    return float(np.mean(specificities)) if specificities else 0.0


def evaluate_model(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "specificity": _specificity_multiclass(y_true_arr, y_pred_arr),
    }