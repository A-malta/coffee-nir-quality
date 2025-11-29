import sys
import os
import pandas as pd
from itertools import product
from tqdm import tqdm
import csv
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_excel_data
from src.models.random_forest import RandomForestModel
from src.evaluation.metrics import evaluate_model

def load_dataset(type_data: str, preprocess_step: str = "007_AreaNorm") -> Tuple[pd.DataFrame, pd.Series]:
    x_path = f"data/processed/{type_data}/{preprocess_step}.xlsx"
    y_path = f"data/raw_split/{type_data}_quality.xlsx"
    
    print(f"Carregando {type_data} ({preprocess_step})...")
    
    X_raw = load_excel_data(x_path)
    y_raw = load_excel_data(y_path)
    
    X = X_raw.iloc[:, 1:].T
    
    # Tenta encontrar a coluna target
    target_col = 'label'
    if target_col not in y_raw.columns:
        # Assume que a segunda coluna é o target (a primeira é o código)
        if len(y_raw.columns) > 1:
            target_col = y_raw.columns[1]
            print(f"Aviso: Coluna 'label' não encontrada. Usando '{target_col}' como target.")
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

def train_evaluate_save(
    params: Dict[str, Any],
    keys: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    csv_file: str
):
    model = RandomForestModel(params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    metrics = evaluate_model(y_val, y_pred)
    
    param_str = "_".join([f"{k}-{v}".replace("None", "NA") for k, v in params.items()])
    model_filename = f"models/rf_{param_str}.joblib"
    model.save(model_filename)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            *[params[k] for k in keys],
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['specificity'],
            model_filename
        ])

def main():
    preprocess_step = "007_AreaNorm"
    
    try:
        X_train, y_train = load_dataset("training", preprocess_step)
        X_val, y_val = load_dataset("validation", preprocess_step)
    except FileNotFoundError as e:
        print(f"Erro crítico: {e}")
        return

    param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
    }
    
    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    
    os.makedirs("models", exist_ok=True)
    csv_file = "resultados_grid_search.csv"
    initialize_csv(csv_file, [*keys, "accuracy", "precision", "recall", "specificity", "model_file"])

    print(f"Iniciando Grid Search com {len(combinations)} combinações...")
    
    for values in tqdm(combinations, ncols=100):
        params = dict(zip(keys, values))
        train_evaluate_save(params, keys, X_train, y_train, X_val, y_val, csv_file)

    print("Grid Search concluído.")

if __name__ == "__main__":
    main()
