import csv
import os
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import infer_class_target_column, load_excel_data
from src.evaluation.metrics import evaluate_model
from src.models.random_forest import RandomForestModel


PROCESSED = Path("data/processed")
RAW_SPLIT = Path("data/raw_split")
MODELS = Path("models")
RESULTS = Path("resultados_grid_search_validacao.csv")
METRICS = ("accuracy", "precision", "recall", "specificity")

PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [300, 500, 800],
    "max_depth": [5, 20, None],
    "min_samples_split": [2, 15],
    "min_samples_leaf": [1, 5, 10],
    "max_features": [None, "sqrt", 0.5, "log2"],
    "bootstrap": [True],
}


def load_dataset(split: str, preprocess: str) -> tuple[pd.DataFrame, pd.Series]:
    X_raw = load_excel_data(str(PROCESSED / split / f"{preprocess}.xlsx"))
    y_raw = load_excel_data(str(RAW_SPLIT / f"{split}_quality.xlsx"))
    X = X_raw.iloc[:, 1:].T.astype(np.float32, copy=False)
    y = y_raw[infer_class_target_column(y_raw)]
    n = min(len(X), len(y))
    return X.iloc[:n], y.iloc[:n]


def model_path(preprocess: str, params: dict[str, Any]) -> Path:
    slug = "_".join(f"{k}-{str(v).replace('None', 'NA')}" for k, v in params.items())
    return MODELS / f"rf_{preprocess}_{slug}.joblib"


def write_header(keys: list[str]) -> None:
    with RESULTS.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["preprocess_step", *keys, *METRICS, "model_file"])


def write_result(preprocess: str, keys: list[str], params: dict[str, Any], metrics: dict[str, float], path: Path) -> None:
    row = [preprocess, *[params[k] for k in keys], *[metrics[m] for m in METRICS], str(path)]
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def train_and_evaluate(params, X_train, y_train, X_val, y_val):
    model = RandomForestModel(params)
    model.fit(X_train, y_train)
    return model, evaluate_model(y_val, model.predict(X_val))


def main() -> None:
    keys = list(PARAM_GRID)
    combinations = list(product(*PARAM_GRID.values()))
    preprocess_files = [p.stem for p in sorted((PROCESSED / "training").glob("*.xlsx"))]

    MODELS.mkdir(exist_ok=True)
    write_header(keys)
    for preprocess in preprocess_files:
        X_train, y_train = load_dataset("training", preprocess)
        X_val, y_val = load_dataset("validation", preprocess)
        for values in combinations:
            params = dict(zip(keys, values))
            model, metrics = train_and_evaluate(params, X_train, y_train, X_val, y_val)
            path = model_path(preprocess, params)
            model.save(str(path))
            write_result(preprocess, keys, params, metrics, path)


if __name__ == "__main__":
    main()
