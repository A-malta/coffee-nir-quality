import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
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


def load_recipe(path):
    with path.open(encoding="utf-8") as f:
        recipe = yaml.safe_load(f)

    model_config = recipe["model"]
    search_config = model_config["search"]
    return search_config


def suggest_params(trial, param_space):
    return {
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
    }


def model_path(preprocess, params, rank):
    slug = "_".join(f"{k}-{str(v).replace('None', 'NA')}" for k, v in params.items())
    return MODELS_DIR / f"rf_{preprocess}_{slug}_rank-{rank:03d}.joblib"


def completed_trials(study):
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def sorted_candidates(candidates):
    if not candidates:
        return []

    study = candidates[0]["study"]
    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE

    return sorted(candidates, key=lambda candidate: candidate["trial"].value, reverse=reverse)


def save_results(results, path):
    results.to_csv(path, index=False)


def feature_selection_table(feature_names, selector):
    return pd.DataFrame(
        {
            WAVELENGTH_COLUMN: feature_names,
            "selected": selector.support_.astype(np.uint8),
        }
    )


def feature_selection_path(preprocess):
    dataset_name = "raw" if preprocess == RAW_PREPROCESS_NAME else f"processed_{preprocess}"
    return DATA_DIR / f"lasso_features_{dataset_name}.xlsx"


def save_feature_selection(preprocess, selection):
    selection.to_excel(feature_selection_path(preprocess), index=False)


def lasso_selector(search_config):
    config = search_config["feature_selection"]["lasso"]
    return LassoFeatureSelector(
        C=float(config["C"]),
        threshold=float(config["threshold"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
    )


def make_model(
    selector,
    rf,
):
    return Pipeline(
        [
            ("lasso_features", selector),
            ("random_forest", rf),
        ]
    )


def score_predictions(y_true, y_pred, scoring):
    if scoring == "min_class_recall":
        return min_class_recall_score(y_true, y_pred)

    return float(evaluate_model(y_true, y_pred)[scoring])


def selected_feature_count(model):
    selector = model.named_steps["lasso_features"]
    return int(selector.n_selected_features_)


def fit_lasso(selector, X_train, y_train):
    selector.fit(X_train, y_train)
    return selector.transform(X_train)


def fit_final_model(params, selector, X_train, y_train):
    X_selected = selector.transform(X_train)

    rf = RandomForestClassifier(**params, n_jobs=-1)
    rf.fit(X_selected, y_train)

    return make_model(selector, rf)


def cross_validation_score(model, X, y, cv, scoring):
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X[test_idx])
        scores.append(score_predictions(y.iloc[test_idx], y_pred, scoring))

    return float(np.mean(scores))


def prepare_search_data(X_train, y_train, search_config):
    cv_folds = int(search_config["cv_folds"])
    search_selector = lasso_selector(search_config)
    X_train_selected = fit_lasso(search_selector, X_train, y_train)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    return search_selector, X_train_selected, cv


def run_optuna_search(preprocess, X_train_selected, y_train, cv, search_config):
    param_space = search_config["space"]
    scoring = search_config["scoring"]
    n_trials = int(search_config["n_trials"])

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction=search_config["direction"], sampler=sampler)

    def objective(trial):
        params = suggest_params(trial, param_space)
        model = RandomForestClassifier(**params, n_jobs=-1)
        return cross_validation_score(model, X_train_selected, y_train, cv, scoring)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with tqdm(total=n_trials, desc=f"  Bayesian Search ({preprocess})", unit="trial", leave=False) as pbar:
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda _study, _trial: pbar.update(1)])

    return study


def build_result_row(preprocess, rank, trial, params, model, metrics, path):
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


def save_top_models(candidates, search_config):
    param_names = list(search_config["space"])
    save_top_models = int(search_config["save_top_models"])
    saved_results = []

    for rank, candidate in enumerate(sorted_candidates(candidates)[:save_top_models], start=1):
        preprocess = candidate["preprocess"]
        trial = candidate["trial"]
        search_selector = candidate["search_selector"]
        X_train = candidate["X_train"]
        y_train = candidate["y_train"]
        params = {name: trial.params[name] for name in param_names}
        model = fit_final_model(params, search_selector, X_train, y_train)

        metrics = evaluate_model(y_train, model.predict(X_train))
        path = model_path(preprocess, params, rank)
        joblib.dump(model, path)

        saved_results.append(build_result_row(preprocess, rank, trial, params, model, metrics, path))

    return saved_results


def optimize_preprocess(preprocess, search_config):
    X_train, y_train = load_modeling_dataset("training", preprocess)
    search_selector, X_train_selected, cv = prepare_search_data(X_train, y_train, search_config)
    selection = feature_selection_table(X_train.columns, search_selector)
    save_feature_selection(preprocess, selection)
    study = run_optuna_search(preprocess, X_train_selected, y_train, cv, search_config)
    return [
        {
            "preprocess": preprocess,
            "study": study,
            "trial": trial,
            "search_selector": search_selector,
            "X_train": X_train,
            "y_train": y_train,
        }
        for trial in completed_trials(study)
    ]


def run_bayesian_search(recipe_file):
    search_config = load_recipe(recipe_file)
    preprocess_files = [RAW_PREPROCESS_NAME, PREPROCESS_NAME]

    MODELS_DIR.mkdir(exist_ok=True)
    candidates = [
        candidate
        for preprocess in preprocess_files
        for candidate in optimize_preprocess(preprocess, search_config)
    ]
    results = save_top_models(candidates, search_config)

    (
        pd.DataFrame(results)
        .sort_values(["preprocess_step", "rank"])
        .pipe(save_results, BAYESIAN_SEARCH_RESULTS_FILE)
    )
