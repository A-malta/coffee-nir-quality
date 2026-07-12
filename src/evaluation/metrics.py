import re
from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def _specificity_multiclass(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calcula a especificidade média para um problema multiclasse.

    Args:
        y_true: Rótulos verdadeiros das amostras.
        y_pred: Rótulos preditos pelo modelo.

    Returns:
        Média das especificidades um-contra-todos das classes válidas.
    """
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


def _slug(value: Any) -> str:
    """Converte um valor em um sufixo seguro para nomes de métricas.

    Args:
        value: Valor que identifica uma classe.

    Returns:
        Texto minúsculo composto por caracteres alfanuméricos e sublinhados.
    """
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    return text.strip("_") or "classe"


def _specificity_by_class(
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    labels: NDArray[Any],
) -> dict[Hashable, float]:
    """Calcula a especificidade um-contra-todos de cada classe.

    Args:
        y_true: Array de rótulos verdadeiros.
        y_pred: Array de rótulos preditos.
        labels: Array ordenado de classes consideradas.

    Returns:
        Mapeamento entre cada classe e sua especificidade.
    """
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


def _accuracy_by_class(
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    labels: NDArray[Any],
) -> dict[Hashable, float]:
    """Calcula a acurácia um-contra-todos de cada classe.

    Args:
        y_true: Array de rótulos verdadeiros.
        y_pred: Array de rótulos preditos.
        labels: Array ordenado de classes consideradas.

    Returns:
        Mapeamento entre cada classe e sua acurácia.
    """
    accuracies = {}

    for label in labels:
        true_positive = np.logical_and(y_true == label, y_pred == label).sum()
        true_negative = np.logical_and(y_true != label, y_pred != label).sum()
        accuracies[label] = float((true_positive + true_negative) / len(y_true))

    return accuracies


def min_class_recall_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Obtém o menor recall observado entre as classes verdadeiras.

    Args:
        y_true: Rótulos verdadeiros das amostras.
        y_pred: Rótulos preditos pelo modelo.

    Returns:
        Menor valor de recall entre todas as classes presentes.
    """
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


def evaluate_model(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Calcula as métricas globais, balanceadas e específicas por classe.

    Args:
        y_true: Rótulos verdadeiros das amostras.
        y_pred: Rótulos preditos pelo modelo.

    Returns:
        Dicionário com métricas agregadas e métricas um-contra-todos por classe.
    """
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
