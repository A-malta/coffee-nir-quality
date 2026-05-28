import csv
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.config import GRID_SEARCH_RESULTS_FILE, MODELS_DIR, RAW_PREPROCESS_NAME
from src.data.dataset import load_modeling_dataset
from src.evaluation.metrics import evaluate_model
from src.preprocessing.spectra import PREPROCESSING_VARIANTS

METRICS = ("accuracy", "precision", "recall", "specificity")


def load_param_grid(path: Path) -> dict[str, list[Any]]:
    with path.open(encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    return recipe["model"]["param_grid"]


def model_path(preprocess: str, params: dict[str, Any]) -> Path:
    slug = "_".join(f"{k}-{str(v).replace('None', 'NA')}" for k, v in params.items())
    return MODELS_DIR / f"rf_{preprocess}_{slug}.joblib"


def write_header(keys: list[str]) -> None:
    with GRID_SEARCH_RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["preprocess_step", *keys, *METRICS, "model_file"])


def write_result(preprocess: str, keys: list[str], params: dict[str, Any], metrics: dict[str, float], path: Path) -> None:
    row = [preprocess, *[params[k] for k in keys], *[metrics[m] for m in METRICS], path.name]
    with GRID_SEARCH_RESULTS_FILE.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def main(recipe_file: Path) -> None:
    param_grid = load_param_grid(recipe_file)
    keys = list(param_grid)
    combinations = list(product(*param_grid.values()))
    preprocess_files = [RAW_PREPROCESS_NAME, *(variant.name for variant in PREPROCESSING_VARIANTS)]
    total = len(preprocess_files) * len(combinations)

    MODELS_DIR.mkdir(exist_ok=True)
    write_header(keys)
    with tqdm(total=total, desc="  Grid Search", unit="combo", leave=False) as pbar:
        for preprocess in preprocess_files:
            X_train, y_train = load_modeling_dataset("training", preprocess)

            for values in combinations:
                params = dict(zip(keys, values))

                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                # Aqui o modelo aprende a classificar os espectros de treino.
                model.fit(X_train, y_train)

                # O modelo prediz as classes dos proprios dados de treino
                # Essas predicoes sao comparadas com y_train para calcular as metricas.

                metrics = evaluate_model(y_train, model.predict(X_train))
                # Calcula as métricas no próprio conjunto de treino

                path = model_path(preprocess, params)
                joblib.dump(model, path)
                write_result(preprocess, keys, params, metrics, path)
                pbar.update(1)
