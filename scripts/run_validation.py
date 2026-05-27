import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_grid_search import RESULTS, load_dataset
from src.evaluation.metrics import evaluate_model
from src.models.random_forest import RandomForestModel


OUT = Path("resultados_validacao_final.csv")
MARKER = "_n_estimators-"


def preprocess_from_model(path: Path) -> str | None:
    stem = path.stem
    return stem[3:stem.index(MARKER)] if stem.startswith("rf_") and MARKER in stem else None


def model_files() -> list[Path]:
    if RESULTS.exists():
        df = pd.read_csv(RESULTS)
        if "model_file" in df:
            return [Path(p) for p in df["model_file"].dropna().drop_duplicates() if Path(p).exists()]
    return sorted(Path("models").glob("*.joblib"))


def evaluate(path: Path) -> dict | None:
    if not re.match(r"rf_.+_n_estimators-", path.name):
        return None
    preprocess = preprocess_from_model(path)
    if preprocess is None:
        return None
    X, y = load_dataset("validation", preprocess)
    metrics = evaluate_model(y, RandomForestModel.load(str(path)).predict(X))
    return {"model": path.name, "preprocess_step": preprocess, **{f"val_{k}": v for k, v in metrics.items()}}


def main() -> None:
    results = [r for path in model_files() if (r := evaluate(path)) is not None]
    if not results:
        print("Nenhum resultado gerado.")
        return
    pd.DataFrame(results).sort_values("val_accuracy", ascending=False).to_csv(OUT, index=False)
    print(f"Validação concluída. Resultados em '{OUT}'.")


if __name__ == "__main__":
    main()
