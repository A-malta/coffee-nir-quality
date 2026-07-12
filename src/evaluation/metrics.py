import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import re


def _specificity_multiclass(y_true, y_pred):
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


def _slug(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    return text.strip("_") or "classe"


def _specificity_by_class(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    specificities = {}

    for idx, label in enumerate(labels):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = total - (tp + fn + fp)
        denom = tn + fp
        specificities[label] = float(tn / denom) if denom > 0 else 0.0

    return specificities


def _accuracy_by_class(y_true, y_pred, labels):
    accuracies = {}

    for label in labels:
        true_positive = np.logical_and(y_true == label, y_pred == label).sum()
        true_negative = np.logical_and(y_true != label, y_pred != label).sum()
        accuracies[label] = float((true_positive + true_negative) / len(y_true))

    return accuracies


def min_class_recall_score(y_true, y_pred):
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = pd.Index(pd.Series(y_true_arr)).unique().to_numpy()
    recalls = recall_score(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return float(np.min(recalls))


def evaluate_model(y_true, y_pred):
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = pd.Index(pd.concat([pd.Series(y_true_arr), pd.Series(y_pred_arr)], ignore_index=True)).unique().to_numpy()
    precision_by_class = precision_score(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )
    recall_by_class = recall_score(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )
    accuracy_by_class = _accuracy_by_class(y_true_arr, y_pred_arr, labels)
    specificity_by_class = _specificity_by_class(y_true_arr, y_pred_arr, labels)
    balanced_precision = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    balanced_recall = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "specificity": _specificity_multiclass(y_true_arr, y_pred_arr),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_precision": float(balanced_precision),
        "balanced_recall": float(balanced_recall),
        "balanced_specificity": _specificity_multiclass(y_true_arr, y_pred_arr),
    }

    for idx, label in enumerate(labels):
        suffix = _slug(label)
        metrics[f"accuracy_{suffix}"] = float(accuracy_by_class[label])
        metrics[f"precision_{suffix}"] = float(precision_by_class[idx])
        metrics[f"recall_{suffix}"] = float(recall_by_class[idx])
        metrics[f"specificity_{suffix}"] = float(specificity_by_class[label])

    return metrics
