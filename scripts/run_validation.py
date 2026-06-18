import joblib
import matplotlib

matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import (
    BAYESIAN_SEARCH_RESULTS_FILE,
    CONFUSION_MATRICES_DIR,
    MODELS_DIR,
    RAW_PREPROCESS_NAME,
    VALIDATION_RESULTS_FILE,
)
from src.data.dataset import load_modeling_dataset
from src.evaluation.metrics import evaluate_model
from src.preprocessing.spectra import PREPROCESS_NAME


MARKER = "_n_estimators-"


def preprocess_from_model(path):
    stem = path.stem
    name = stem.removeprefix("rf_")
    preprocess_names = [RAW_PREPROCESS_NAME, PREPROCESS_NAME]

    for preprocess in sorted(preprocess_names, key=len, reverse=True):
        if name == preprocess or name.startswith(f"{preprocess}_"):
            return preprocess

    return name.split(MARKER, 1)[0]


def model_files():
    results = pd.read_csv(BAYESIAN_SEARCH_RESULTS_FILE)
    return [MODELS_DIR / name for name in results["model_file"].dropna()]


def save_csv(results, path):
    results.to_csv(path, index=False)


def plot_confusion_matrix(path, y_true, y_pred, final_rank):
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
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Prevista")
    plt.savefig(CONFUSION_MATRICES_DIR / f"rank-final-{final_rank:03d}_{path.stem}_cm.png", dpi=300)
    plt.close()


def evaluate(path, datasets):
    preprocess = preprocess_from_model(path)

    if preprocess not in datasets:
        datasets[preprocess] = load_modeling_dataset("validation", preprocess)
    X, y = datasets[preprocess]

    model = joblib.load(path)

    y_pred = model.predict(X)
    metrics = evaluate_model(y, y_pred)

    return {
        "model": path.name,
        "preprocess_step": preprocess,
        **{f"val_{k}": v for k, v in metrics.items()},
    }


def sort_results(results):
    recall_columns = [column for column in results.columns if column.startswith("val_recall_")]
    if not recall_columns:
        return results.sort_values("val_accuracy", ascending=False)

    return (
        results.assign(
            _min_class_recall=results[recall_columns].min(axis=1),
            _mean_class_recall=results[recall_columns].mean(axis=1),
        )
        .sort_values(
            ["_min_class_recall", "_mean_class_recall", "val_accuracy"],
            ascending=False,
        )
        .drop(columns=["_min_class_recall", "_mean_class_recall"])
    )


def add_final_rank(results):
    results = results.reset_index(drop=True).copy()
    results["rank"] = range(1, len(results) + 1)
    return results


def plot_validation_confusion_matrices(validation_results, datasets):
    for final_rank, model_name in enumerate(validation_results["model"], start=1):
        path = MODELS_DIR / model_name
        preprocess = preprocess_from_model(path)

        if preprocess not in datasets:
            datasets[preprocess] = load_modeling_dataset("validation", preprocess)
        X, y = datasets[preprocess]

        model = joblib.load(path)
        plot_confusion_matrix(path, y, model.predict(X), final_rank)


def run_validation():
    CONFUSION_MATRICES_DIR.mkdir(exist_ok=True)
    datasets = {}
    results = [evaluate(path, datasets) for path in model_files()]
    validation_results = add_final_rank(sort_results(pd.DataFrame(results)))

    save_csv(validation_results, VALIDATION_RESULTS_FILE)
    plot_validation_confusion_matrices(validation_results, datasets)

    print(f"Pipeline concluída. Resultados em '{VALIDATION_RESULTS_FILE}'.")
