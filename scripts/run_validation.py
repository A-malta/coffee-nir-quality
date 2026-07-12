from pathlib import Path
from typing import TypedDict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from numpy.typing import ArrayLike
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


class ValidationCandidate(TypedDict):
    """Metadados de um modelo candidato à validação final."""

    path: Path
    cv_rank: int
    cv_score: float


Dataset = tuple[pd.DataFrame, pd.Series]
Datasets = dict[str, Dataset]
EvaluationResult = dict[str, str | int | float]


def preprocess_from_model(path: Path) -> str:
    """Extrai o nome do pré-processamento do arquivo de um modelo.

    Args:
        path: Caminho do arquivo serializado do modelo.

    Returns:
        Nome da etapa de pré-processamento associada ao modelo.
    """
    stem = path.stem
    name = stem.removeprefix("rf_")
    preprocess_names = [RAW_PREPROCESS_NAME, PREPROCESS_NAME]

    for preprocess in sorted(preprocess_names, key=len, reverse=True):
        if name == preprocess or name.startswith(f"{preprocess}_"):
            return preprocess

    return name.split(MARKER, 1)[0]


def validation_model_limit(recipe_file: Path) -> int:
    """Obtém da recipe o limite de modelos usados na validação final.

    Args:
        recipe_file: Caminho do arquivo YAML da recipe.

    Returns:
        Quantidade máxima de modelos a validar.

    Raises:
        FileNotFoundError: Se o arquivo da recipe não existir.
        KeyError: Se a configuração esperada não estiver presente na recipe.
    """
    with recipe_file.open(encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    return int(recipe["model"]["search"]["validate_top_models"])


def validation_candidates(limit: int) -> list[ValidationCandidate]:
    """Seleciona os melhores candidatos segundo a validação cruzada.

    Args:
        limit: Número máximo de candidatos a selecionar.

    Returns:
        Lista com caminho, posição e pontuação de cada candidato.

    Raises:
        FileNotFoundError: Se o CSV de resultados da busca não existir.
    """
    results = pd.read_csv(BAYESIAN_SEARCH_RESULTS_FILE)
    selected = (
        results.dropna(subset=["model_file", "cv_score", "rank"])
        .sort_values(["cv_score", "rank"], ascending=[False, True])
        .head(limit)
    )
    return [
        {
            "path": MODELS_DIR / row.model_file,
            "cv_rank": int(row.rank),
            "cv_score": float(row.cv_score),
        }
        for row in selected.itertuples(index=False)
    ]


def save_csv(results: pd.DataFrame, path: str | Path) -> None:
    """Salva uma tabela de resultados em formato CSV.

    Args:
        results: Tabela que será salva.
        path: Caminho do arquivo CSV de destino.
    """
    results.to_csv(path, index=False)


def plot_confusion_matrix(
    path: Path,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    final_rank: int,
) -> None:
    """Gera a matriz de confusão percentual de um modelo.

    Args:
        path: Caminho do arquivo do modelo avaliado.
        y_true: Classes reais do conjunto de validação.
        y_pred: Classes previstas pelo modelo.
        final_rank: Posição final do modelo na validação.
    """
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


def evaluate(candidate: ValidationCandidate, datasets: Datasets) -> EvaluationResult:
    """Avalia um modelo candidato no conjunto de validação externa.

    Args:
        candidate: Metadados do modelo que será avaliado.
        datasets: Cache dos conjuntos de dados por pré-processamento.

    Returns:
        Métricas e metadados obtidos na avaliação do modelo.

    Raises:
        FileNotFoundError: Se o modelo ou os dados necessários não existirem.
    """
    path = candidate["path"]
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
        "cv_rank": candidate["cv_rank"],
        "cv_score": candidate["cv_score"],
        **{f"val_{k}": v for k, v in metrics.items()},
    }


def sort_results(results: pd.DataFrame) -> pd.DataFrame:
    """Ordena os modelos pelas métricas de validação prioritárias.

    Args:
        results: Tabela com as métricas dos modelos avaliados.

    Returns:
        Tabela ordenada por recall de classe e acurácia.
    """
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


def add_final_rank(results: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a posição final sequencial aos resultados ordenados.

    Args:
        results: Tabela de resultados já ordenada.

    Returns:
        Cópia da tabela com a coluna de posição final.
    """
    results = results.reset_index(drop=True).copy()
    results["rank"] = range(1, len(results) + 1)
    return results


def plot_validation_confusion_matrices(
    validation_results: pd.DataFrame,
    datasets: Datasets,
) -> None:
    """Gera as matrizes de confusão dos modelos validados.

    Args:
        validation_results: Resultados finais com os nomes dos modelos.
        datasets: Cache dos conjuntos de validação por pré-processamento.
    """
    for final_rank, model_name in enumerate(validation_results["model"], start=1):
        path = MODELS_DIR / model_name
        preprocess = preprocess_from_model(path)

        if preprocess not in datasets:
            datasets[preprocess] = load_modeling_dataset("validation", preprocess)
        X, y = datasets[preprocess]

        model = joblib.load(path)
        plot_confusion_matrix(path, y, model.predict(X), final_rank)


def clear_generated_confusion_matrices() -> None:
    """Remove matrizes de confusão geradas por execuções anteriores."""
    if not CONFUSION_MATRICES_DIR.exists():
        return

    for path in CONFUSION_MATRICES_DIR.glob("rank-final-*_rf_*_cm.png"):
        if path.is_file():
            path.unlink()


def run_validation(recipe_file: Path) -> None:
    """Executa a validação externa dos melhores modelos da busca.

    Args:
        recipe_file: Caminho do arquivo YAML que configura a pipeline.

    Raises:
        FileNotFoundError: Se a recipe, os resultados, modelos ou dados não existirem.
    """
    clear_generated_confusion_matrices()
    CONFUSION_MATRICES_DIR.mkdir(exist_ok=True)
    datasets = {}
    limit = validation_model_limit(recipe_file)
    results = [evaluate(candidate, datasets) for candidate in validation_candidates(limit)]
    validation_results = add_final_rank(sort_results(pd.DataFrame(results)))

    save_csv(validation_results, VALIDATION_RESULTS_FILE)
    plot_validation_confusion_matrices(validation_results, datasets)

    print(f"Pipeline concluída. Resultados em '{VALIDATION_RESULTS_FILE}'.")
