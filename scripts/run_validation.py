from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import (
    CONFUSION_MATRICES_DIR,
    MODELS_DIR,
    VALIDATION_RESULTS_FILE,
)
from src.data.dataset import load_modeling_dataset
from src.evaluation.metrics import evaluate_model


MARKER = "_n_estimators-"


def preprocess_from_model(path: Path) -> str:
    stem = path.stem
    return stem.removeprefix("rf_").split(MARKER, 1)[0]


def model_files() -> list[Path]:
    return sorted(MODELS_DIR.glob("*.joblib"))


def plot_confusion_matrix(path: Path, y_true, y_pred) -> None:
    labels = (
        pd.Index(pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True).dropna())
        .unique()
        .tolist()
    )
    cm_percentage = confusion_matrix(y_true, y_pred, labels=labels, normalize="true") * 100

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Porcentagem (%)"},
        vmin=0,
        vmax=100,
    )
    plt.title(f"Matriz de Confusão\n{path.stem}", fontsize=10)
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Prevista")
    plt.savefig(CONFUSION_MATRICES_DIR / f"{path.stem}_cm.png", dpi=300)
    plt.close()


def evaluate(
    path: Path,
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> dict[str, float | str]:
    preprocess = preprocess_from_model(path)

    if preprocess not in datasets:
        datasets[preprocess] = load_modeling_dataset("validation", preprocess)
    X, y = datasets[preprocess]

    model = joblib.load(path)

    y_pred = model.predict(X)
    metrics = evaluate_model(y, y_pred)

    plot_confusion_matrix(path, y, y_pred)

    return {
        "model": path.name,
        "preprocess_step": preprocess,
        **{f"val_{k}": v for k, v in metrics.items()},
    }


def main() -> None:
    CONFUSION_MATRICES_DIR.mkdir(exist_ok=True)
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    results = [evaluate(path, datasets) for path in model_files()]
    (
        pd.DataFrame(results)
        .sort_values("val_accuracy", ascending=False)
        .to_csv(VALIDATION_RESULTS_FILE, index=False)
    )
    print(f"Pipeline concluída. Resultados em '{VALIDATION_RESULTS_FILE}'.")
