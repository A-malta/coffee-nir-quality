import sys
import os
import pandas as pd
from itertools import product
from tqdm import tqdm
import csv
from typing import List, Dict, Any, Tuple
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_excel_data
from src.models.random_forest import RandomForestModel
from src.evaluation.metrics import evaluate_model

def load_dataset(type_data: str, preprocess_step: str) -> Tuple[pd.DataFrame, pd.Series]:
    x_path = f"data/processed/{type_data}/{preprocess_step}.xlsx"
    y_path = f"data/raw_split/{type_data}_quality.xlsx"
    
    X_raw = load_excel_data(x_path)
    y_raw = load_excel_data(y_path)
    
    X = X_raw.iloc[:, 1:].T
    
    target_col = 'label'
    if target_col not in y_raw.columns:
        if len(y_raw.columns) > 1:
            target_col = y_raw.columns[1]
        else:
            raise KeyError("Não foi possível identificar a coluna de target (label).")
            
    y = y_raw[target_col]
    
    min_size = min(X.shape[0], len(y))
    return X.iloc[:min_size], y.iloc[:min_size]

def initialize_csv(filepath: str, headers: List[str]):
    if not os.path.exists(filepath):
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def save_metrics(csv_file: str, preprocess_step: str, params: Dict[str, Any], keys: List[str], metrics: Dict[str, float], model_filename: str):
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            preprocess_step,
            *[params[k] for k in keys],
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['specificity'],
            model_filename
        ])

def train_evaluate_save(
    params: Dict[str, Any],
    keys: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    csv_val: str,
    csv_test: str,
    preprocess_step: str
):
    model = RandomForestModel(params)
    model.fit(X_train, y_train)
    

    y_pred_val = model.predict(X_val)
    metrics_val = evaluate_model(y_val, y_pred_val)
    

    y_pred_test = model.predict(X_test)
    metrics_test = evaluate_model(y_test, y_pred_test)
    
    param_str = "_".join([f"{k}-{v}".replace("None", "NA") for k, v in params.items()])
    model_filename = f"models/rf_{preprocess_step}_{param_str}.joblib"
    model.save(model_filename)
    
    save_metrics(csv_val, preprocess_step, params, keys, metrics_val, model_filename)
    save_metrics(csv_test, preprocess_step, params, keys, metrics_test, model_filename)

def get_preprocess_files() -> List[str]:
    path = "data/processed/training"
    if not os.path.exists(path):
        return []
    files = glob.glob(os.path.join(path, "*.xlsx"))
    return [os.path.splitext(os.path.basename(f))[0] for f in sorted(files)]

def main():
    preprocess_files = get_preprocess_files()
    
    if not preprocess_files:
        print("Nenhum arquivo de pré-processamento encontrado em data/processed/training/")
        return

    param_grid = {
        'n_estimators': [50, 100, 200, 300], 
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [None],
        'bootstrap': [True, False],
    }
    
    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    
    os.makedirs("models", exist_ok=True)
    csv_val = "resultados_grid_search_validacao.csv"
    csv_test = "resultados_grid_search_teste.csv"
    
    headers = ["preprocess_step", *keys, "accuracy", "precision", "recall", "specificity", "model_file"]
    initialize_csv(csv_val, headers)
    initialize_csv(csv_test, headers)

    total_steps = len(preprocess_files) * len(combinations)
    print(f"Iniciando Grid Search: {len(preprocess_files)} pré-processamentos x {len(combinations)} combinações = {total_steps} modelos.")
    
    for preprocess_step in preprocess_files:
        print(f"\nProcessando: {preprocess_step}")
        try:
            X_train, y_train = load_dataset("training", preprocess_step)
            X_val, y_val = load_dataset("validation", preprocess_step)
            X_test, y_test = load_dataset("test", preprocess_step)
            
            for values in tqdm(combinations, ncols=100, desc=preprocess_step):
                params = dict(zip(keys, values))
                train_evaluate_save(params, keys, X_train, y_train, X_val, y_val, X_test, y_test, csv_val, csv_test, preprocess_step)
                
        except Exception as e:
            print(f"Erro ao processar {preprocess_step}: {e}")

    print("Grid Search concluído.")

if __name__ == "__main__":
    main()
