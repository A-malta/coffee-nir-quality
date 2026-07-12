from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict, cast

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.config import (
    BAYESIAN_SEARCH_RESULTS_FILE,
    DATA_DIR,
    MODELS_DIR,
    RAW_PREPROCESS_NAME,
    WAVELENGTH_COLUMN,
)
from src.data.dataset import load_modeling_dataset
from src.evaluation.metrics import evaluate_model, min_class_recall_score
from src.modeling.feature_selection import LassoFeatureSelector
from src.preprocessing.spectra import PREPROCESS_NAME


class IntegerRange(TypedDict):
    """Intervalo discreto de um hiperparâmetro inteiro."""

    low: int
    high: int
    step: int


class FloatRange(TypedDict):
    """Intervalo contínuo de um hiperparâmetro real."""

    low: float
    high: float
    distribution: str


class ParameterSpace(TypedDict):
    """Espaço de busca dos hiperparâmetros da floresta aleatória."""

    n_estimators: IntegerRange
    max_depth: IntegerRange
    min_samples_split: IntegerRange
    min_samples_leaf: IntegerRange
    max_features: FloatRange
    bootstrap: list[bool]


class LassoConfig(TypedDict):
    """Configuração do seletor de variáveis com penalização L1."""

    enabled: bool
    standardize: bool
    penalty: str
    solver: str
    C: float
    threshold: float
    max_iter: int
    tol: float


class FeatureSelectionConfig(TypedDict):
    """Configuração das estratégias de seleção de variáveis."""

    lasso: LassoConfig


class SearchConfig(TypedDict):
    """Configuração executável da busca Bayesiana."""

    algorithm: str
    sampler: str
    n_trials: int
    cv_folds: int
    cv_shuffle: bool
    scoring: str
    direction: Literal["maximize", "minimize"]
    save_top_models: int
    validate_top_models: int
    space: ParameterSpace
    feature_selection: FeatureSelectionConfig


class ModelConfig(TypedDict):
    """Trecho da receita dedicado à modelagem."""

    search: SearchConfig


class RecipeConfig(TypedDict):
    """Estrutura mínima da receita consumida por este módulo."""

    model: ModelConfig


class RandomForestParams(TypedDict):
    """Hiperparâmetros sorteados para uma floresta aleatória."""

    n_estimators: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    max_features: float
    bootstrap: bool


class Candidate(TypedDict):
    """Candidato completo produzido por uma busca de pré-processamento."""

    preprocess: str
    study: optuna.study.Study
    trial: optuna.trial.FrozenTrial
    final_selector: LassoFeatureSelector
    X_train: pd.DataFrame
    y_train: pd.Series


ParameterValue: TypeAlias = int | float | bool
CVSplit: TypeAlias = tuple[NDArray[np.intp], NDArray[np.intp]]
ResultRow: TypeAlias = dict[str, object]


def load_recipe(path: Path) -> SearchConfig:
    """Carrega a configuração da busca presente em uma receita YAML.

    Args:
        path: Caminho do arquivo de receita.

    Returns:
        Configuração da busca Bayesiana definida na receita.

    Raises:
        FileNotFoundError: Se o arquivo de receita não existir.
        KeyError: Se a receita não contiver as seções ``model`` e ``search``.
    """
    with path.open(encoding="utf-8") as f:
        recipe = cast(RecipeConfig, yaml.safe_load(f))

    model_config = recipe["model"]
    search_config = model_config["search"]
    return search_config


def suggest_params(trial: optuna.trial.Trial, param_space: ParameterSpace) -> RandomForestParams:
    """Solicita ao Optuna um conjunto de hiperparâmetros da floresta.

    Args:
        trial: Tentativa ativa do estudo Optuna.
        param_space: Limites e opções do espaço de busca.

    Returns:
        Hiperparâmetros sugeridos para a tentativa.
    """
    return cast(RandomForestParams, {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            int(param_space["n_estimators"]["low"]),
            int(param_space["n_estimators"]["high"]),
            step=int(param_space["n_estimators"]["step"]),
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            int(param_space["max_depth"]["low"]),
            int(param_space["max_depth"]["high"]),
            step=int(param_space["max_depth"]["step"]),
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            int(param_space["min_samples_split"]["low"]),
            int(param_space["min_samples_split"]["high"]),
            step=int(param_space["min_samples_split"]["step"]),
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            int(param_space["min_samples_leaf"]["low"]),
            int(param_space["min_samples_leaf"]["high"]),
            step=int(param_space["min_samples_leaf"]["step"]),
        ),
        "max_features": trial.suggest_float(
            "max_features",
            float(param_space["max_features"]["low"]),
            float(param_space["max_features"]["high"]),
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", param_space["bootstrap"]),
    })


def model_path(preprocess: str, params: Mapping[str, ParameterValue], rank: int) -> Path:
    """Monta o caminho de persistência de um modelo ranqueado.

    Args:
        preprocess: Nome do pré-processamento aplicado aos espectros.
        params: Hiperparâmetros usados pelo modelo.
        rank: Posição do modelo no ranking dos candidatos.

    Returns:
        Caminho do arquivo Joblib que armazenará o modelo.
    """
    slug = "_".join(f"{k}-{str(v).replace('None', 'NA')}" for k, v in params.items())
    return MODELS_DIR / f"rf_{preprocess}_{slug}_rank-{rank:03d}.joblib"


def completed_trials(study: optuna.study.Study) -> list[optuna.trial.FrozenTrial]:
    """Seleciona as tentativas concluídas e dotadas de valor objetivo.

    Args:
        study: Estudo Optuna a ser inspecionado.

    Returns:
        Tentativas concluídas que produziram uma pontuação.
    """
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def sorted_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Ordena candidatos conforme a direção de otimização do estudo.

    Args:
        candidates: Candidatos gerados pelas buscas de pré-processamento.

    Returns:
        Nova lista ordenada do melhor para o pior candidato.
    """
    if not candidates:
        return []

    study = candidates[0]["study"]
    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE

    return sorted(
        candidates,
        key=lambda candidate: cast(float, candidate["trial"].value),
        reverse=reverse,
    )


def save_results(results: pd.DataFrame, path: Path) -> None:
    """Salva a tabela consolidada de resultados em CSV.

    Args:
        results: Tabela com métricas, parâmetros e caminhos dos modelos.
        path: Caminho do arquivo CSV de destino.
    """
    results.to_csv(path, index=False)


def feature_selection_table(feature_names: pd.Index, selector: LassoFeatureSelector) -> pd.DataFrame:
    """Cria uma tabela que identifica as variáveis selecionadas pelo LASSO.

    Args:
        feature_names: Nomes ou comprimentos de onda das variáveis de entrada.
        selector: Seletor LASSO já ajustado.

    Returns:
        Tabela com cada variável e seu indicador binário de seleção.
    """
    return pd.DataFrame(
        {
            WAVELENGTH_COLUMN: feature_names,
            "selected": selector.support_.astype(np.uint8),
        }
    )


def feature_selection_path(preprocess: str) -> Path:
    """Define o caminho da planilha de variáveis selecionadas.

    Args:
        preprocess: Nome do pré-processamento aplicado aos espectros.

    Returns:
        Caminho da planilha Excel correspondente.
    """
    dataset_name = "raw" if preprocess == RAW_PREPROCESS_NAME else f"processed_{preprocess}"
    return DATA_DIR / f"lasso_features_{dataset_name}.xlsx"


def save_feature_selection(preprocess: str, selection: pd.DataFrame) -> None:
    """Persiste a seleção de variáveis de um pré-processamento.

    Args:
        preprocess: Nome do pré-processamento aplicado aos espectros.
        selection: Tabela das variáveis e dos respectivos indicadores.
    """
    selection.to_excel(feature_selection_path(preprocess), index=False)


def lasso_selector(search_config: SearchConfig) -> LassoFeatureSelector:
    """Constrói o seletor LASSO especificado pela receita.

    Args:
        search_config: Configuração completa da busca Bayesiana.

    Returns:
        Seletor de variáveis ainda não ajustado.

    Raises:
        ValueError: Se a seleção de variáveis LASSO estiver desabilitada.
    """
    config = search_config["feature_selection"]["lasso"]
    if not config.get("enabled", False):
        raise ValueError("A recipe do TCC exige seleção de variáveis LASSO habilitada.")

    return LassoFeatureSelector(
        C=float(config["C"]),
        threshold=float(config["threshold"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        standardize=bool(config.get("standardize", True)),
        penalty=config.get("penalty", "l1"),
        solver=config.get("solver", "saga"),
    )


def make_model(
    selector: LassoFeatureSelector,
    rf: RandomForestClassifier,
) -> Pipeline:
    """Combina o seletor de variáveis e a floresta em um pipeline.

    Args:
        selector: Transformador responsável pela seleção de variáveis.
        rf: Classificador de floresta aleatória.

    Returns:
        Pipeline de seleção de variáveis e classificação.
    """
    return Pipeline(
        [
            ("lasso_features", selector),
            ("random_forest", rf),
        ]
    )


def score_predictions(y_true: ArrayLike, y_pred: ArrayLike, scoring: str) -> float:
    """Calcula a métrica configurada para um conjunto de predições.

    Args:
        y_true: Rótulos verdadeiros das amostras.
        y_pred: Rótulos preditos pelo modelo.
        scoring: Nome da métrica a ser calculada.

    Returns:
        Valor escalar da métrica solicitada.

    Raises:
        KeyError: Se ``scoring`` não for uma métrica produzida pela avaliação.
    """
    if scoring == "min_class_recall":
        return min_class_recall_score(y_true, y_pred)

    return float(evaluate_model(y_true, y_pred)[scoring])


def selected_feature_count(model: Pipeline) -> int:
    """Obtém o número de variáveis mantidas por um pipeline ajustado.

    Args:
        model: Pipeline que contém o passo ``lasso_features`` ajustado.

    Returns:
        Quantidade de variáveis selecionadas.
    """
    selector = cast(LassoFeatureSelector, model.named_steps["lasso_features"])
    return int(selector.n_selected_features_)


def modeling_pipeline(params: RandomForestParams, search_config: SearchConfig) -> Pipeline:
    """Cria um pipeline não ajustado para uma tentativa da busca.

    Args:
        params: Hiperparâmetros da floresta aleatória.
        search_config: Configuração completa da busca Bayesiana.

    Returns:
        Pipeline pronto para ser ajustado dentro da validação cruzada.
    """
    selector = lasso_selector(search_config)
    rf = RandomForestClassifier(**params, n_jobs=-1)
    return make_model(selector, rf)


def fit_final_model(
    params: RandomForestParams,
    selector: LassoFeatureSelector,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Ajusta a floresta final sobre as variáveis previamente selecionadas.

    Args:
        params: Hiperparâmetros da floresta aleatória.
        selector: Seletor LASSO já ajustado com o treino completo.
        X_train: Matriz de variáveis do conjunto de treinamento.
        y_train: Rótulos do conjunto de treinamento.

    Returns:
        Pipeline com seletor e floresta ajustados.
    """
    X_selected = selector.transform(X_train)
    rf = RandomForestClassifier(**params, n_jobs=-1)
    rf.fit(X_selected, y_train)
    return make_model(selector, rf)


def cross_validation_score(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: Sequence[CVSplit],
    scoring: str,
) -> float:
    """Calcula a pontuação média de um modelo por validação cruzada.

    Args:
        model: Pipeline não ajustado a ser clonado em cada dobra.
        X: Matriz de variáveis do conjunto de treinamento.
        y: Rótulos do conjunto de treinamento.
        cv_splits: Índices de treino e teste de cada dobra.
        scoring: Nome da métrica usada para pontuar as predições.

    Returns:
        Média das pontuações obtidas nas dobras.
    """
    X_array = np.asarray(X, dtype=np.float32)
    scores = []
    for train_idx, test_idx in cv_splits:
        fold_model = clone(model)
        fold_model.fit(X_array[train_idx], y.iloc[train_idx])
        y_pred = fold_model.predict(X_array[test_idx])
        scores.append(score_predictions(y.iloc[test_idx], y_pred, scoring))

    return float(np.mean(scores))


def prepare_cross_validation_splits(
    X: pd.DataFrame,
    y: pd.Series,
    search_config: SearchConfig,
) -> list[CVSplit]:
    """Gera uma única partição estratificada reutilizável pela busca.

    Args:
        X: Matriz de variáveis do conjunto de treinamento.
        y: Rótulos usados na estratificação das dobras.
        search_config: Configuração completa da busca Bayesiana.

    Returns:
        Pares de índices de treino e teste para cada dobra.

    Raises:
        ValueError: Se os dados não permitirem o número configurado de dobras.
    """
    cv_folds = int(search_config["cv_folds"])
    cv_shuffle = bool(search_config.get("cv_shuffle", True))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=cv_shuffle)
    return [(train_idx.copy(), test_idx.copy()) for train_idx, test_idx in cv.split(X, y)]


def run_optuna_search(
    preprocess: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: Sequence[CVSplit],
    search_config: SearchConfig,
) -> optuna.study.Study:
    """Executa a otimização Bayesiana para um pré-processamento.

    Args:
        preprocess: Nome do pré-processamento avaliado.
        X_train: Matriz de variáveis do conjunto de treinamento.
        y_train: Rótulos do conjunto de treinamento.
        cv_splits: Índices fixos das dobras de validação cruzada.
        search_config: Configuração completa da busca Bayesiana.

    Returns:
        Estudo Optuna com todas as tentativas executadas.
    """
    param_space = search_config["space"]
    scoring = search_config["scoring"]
    n_trials = int(search_config["n_trials"])

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction=search_config["direction"], sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        """Avalia uma tentativa por validação cruzada.

        Args:
            trial: Tentativa ativa que fornecerá os hiperparâmetros.

        Returns:
            Pontuação média da tentativa nas dobras configuradas.
        """
        params = suggest_params(trial, param_space)
        model = modeling_pipeline(params, search_config)
        return cross_validation_score(model, X_train, y_train, cv_splits, scoring)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with tqdm(total=n_trials, desc=f"  Bayesian Search ({preprocess})", unit="trial", leave=False) as pbar:
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda _study, _trial: pbar.update(1)])

    return study


def build_result_row(
    preprocess: str,
    rank: int,
    trial: optuna.trial.FrozenTrial,
    params: RandomForestParams,
    model: Pipeline,
    metrics: Mapping[str, float],
    path: Path,
) -> ResultRow:
    """Consolida metadados e métricas de um modelo salvo.

    Args:
        preprocess: Nome do pré-processamento avaliado.
        rank: Posição do modelo entre os melhores candidatos.
        trial: Tentativa Optuna que originou o modelo.
        params: Hiperparâmetros usados pela floresta aleatória.
        model: Pipeline final ajustado.
        metrics: Métricas calculadas no conjunto de treinamento.
        path: Caminho em que o modelo foi persistido.

    Returns:
        Registro heterogêneo pronto para compor a tabela de resultados.
    """
    return {
        "preprocess_step": preprocess,
        "rank": rank,
        "cv_score": trial.value,
        "feature_selector": "lasso",
        "selected_features": selected_feature_count(model),
        **params,
        **{f"train_{metric}": value for metric, value in metrics.items()},
        "model_file": path.name,
    }


def save_top_models(
    candidates: Sequence[Candidate],
    search_config: SearchConfig,
) -> list[ResultRow]:
    """Ajusta, persiste e descreve os melhores candidatos da busca.

    Args:
        candidates: Candidatos concluídos de todos os pré-processamentos.
        search_config: Configuração completa da busca Bayesiana.

    Returns:
        Registros dos modelos finais salvos, em ordem de ranking.
    """
    param_names = list(search_config["space"])
    save_top_models = int(search_config["save_top_models"])
    saved_results: list[ResultRow] = []

    for rank, candidate in enumerate(sorted_candidates(candidates)[:save_top_models], start=1):
        preprocess = candidate["preprocess"]
        trial = candidate["trial"]
        final_selector = candidate["final_selector"]
        X_train = candidate["X_train"]
        y_train = candidate["y_train"]
        params = cast(RandomForestParams, {name: trial.params[name] for name in param_names})
        model = fit_final_model(params, final_selector, X_train, y_train)

        metrics = evaluate_model(y_train, model.predict(X_train))
        path = model_path(preprocess, params, rank)
        joblib.dump(model, path)

        saved_results.append(build_result_row(preprocess, rank, trial, params, model, metrics, path))

    return saved_results


def optimize_preprocess(
    preprocess: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: Sequence[CVSplit],
    search_config: SearchConfig,
) -> list[Candidate]:
    """Otimiza modelos e registra a seleção final de um pré-processamento.

    Args:
        preprocess: Nome do pré-processamento avaliado.
        X_train: Matriz de variáveis do conjunto de treinamento.
        y_train: Rótulos do conjunto de treinamento.
        cv_splits: Índices fixos das dobras de validação cruzada.
        search_config: Configuração completa da busca Bayesiana.

    Returns:
        Candidatos associados às tentativas concluídas do estudo.
    """
    study = run_optuna_search(preprocess, X_train, y_train, cv_splits, search_config)
    final_selector = lasso_selector(search_config).fit(X_train, y_train)
    selection = feature_selection_table(X_train.columns, final_selector)
    save_feature_selection(preprocess, selection)
    return [
        {
            "preprocess": preprocess,
            "study": study,
            "trial": trial,
            "final_selector": final_selector,
            "X_train": X_train,
            "y_train": y_train,
        }
        for trial in completed_trials(study)
    ]


def clear_generated_models() -> None:
    """Remove modelos ranqueados gerados por execuções anteriores."""
    if not MODELS_DIR.exists():
        return

    for path in MODELS_DIR.glob("rf_*_rank-*.joblib"):
        if path.is_file():
            path.unlink()


def run_bayesian_search(recipe_file: Path) -> None:
    """Executa a busca Bayesiana nos conjuntos bruto e pré-processado.

    Args:
        recipe_file: Caminho da receita YAML que orienta a busca.

    Raises:
        ValueError: Se os conjuntos de treinamento estiverem desalinhados.
    """
    search_config = load_recipe(recipe_file)
    preprocess_files = [RAW_PREPROCESS_NAME, PREPROCESS_NAME]
    training_datasets = {
        preprocess: load_modeling_dataset("training", preprocess)
        for preprocess in preprocess_files
    }

    reference_X, reference_y = training_datasets[preprocess_files[0]]
    for preprocess, (X_train, y_train) in training_datasets.items():
        if len(X_train) != len(reference_X) or not np.array_equal(np.asarray(y_train), np.asarray(reference_y)):
            raise ValueError(f"Dados de treinamento desalinhados para o pré-processamento '{preprocess}'.")

    cv_splits = prepare_cross_validation_splits(reference_X, reference_y, search_config)

    clear_generated_models()
    MODELS_DIR.mkdir(exist_ok=True)
    candidates = [
        candidate
        for preprocess in preprocess_files
        for candidate in optimize_preprocess(
            preprocess,
            *training_datasets[preprocess],
            cv_splits,
            search_config,
        )
    ]
    results = save_top_models(candidates, search_config)

    (
        pd.DataFrame(results)
        .sort_values(["preprocess_step", "rank"])
        .pipe(save_results, BAYESIAN_SEARCH_RESULTS_FILE)
    )
