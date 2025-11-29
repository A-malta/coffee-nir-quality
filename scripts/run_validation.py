import sys
import os
import pandas as pd
import glob
from typing import Tuple, Dict, Any, Optional
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_excel_data
from src.models.random_forest import RandomForestModel
from src.evaluation.metrics import evaluate_model

def load_dataset(type_data: str, preprocess_step: str) -> Tuple[pd.DataFrame, pd.Series]:
    x_path = f"data/processed/{type_data}/{preprocess_step}.xlsx"
    y_path = f"data/raw_split/{type_data}_quality.xlsx"
    
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {x_path}")

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

def extract_preprocess_step(model_filename: str) -> Optional[str]:
    match = re.search(r"rf_(\d{3}_.+?)_n_estimators", model_filename)
    if match:
        return match.group(1)
    return None

def evaluate_single_model(model_path: str) -> Optional[Dict[str, Any]]:
    try:
        filename = os.path.basename(model_path)
        
        if not re.match(r"rf_\d{3}_", filename):
            return None
            
        preprocess_step = extract_preprocess_step(filename)
        
        if not preprocess_step:
            print(f"Aviso: Não foi possível identificar o pré-processamento no arquivo {filename}. Pulando.")
            return None

        X_test, y_test = load_dataset("test", preprocess_step)
        X_val, y_val = load_dataset("validation", preprocess_step)
        
        model = RandomForestModel.load(model_path)
        
        y_pred_test = model.predict(X_test)
        metrics_test = evaluate_model(y_test, y_pred_test)
        
        y_pred_val = model.predict(X_val)
        metrics_val = evaluate_model(y_val, y_pred_val)
        
        result = {
            "model": filename,
            "preprocess_step": preprocess_step,
        }
        
        for k, v in metrics_test.items():
            result[f"test_{k}"] = v
            
        for k, v in metrics_val.items():
            result[f"val_{k}"] = v
            
        return result
    except Exception as e:
        print(f"Erro ao avaliar {model_path}: {e}")
        return None

def main():
    models_dir = "models"
    model_files = glob.glob(os.path.join(models_dir, "*.joblib"))
    
    if not model_files:
        print(f"Nenhum modelo encontrado em {models_dir}")
        return

    print(f"Avaliando {len(model_files)} modelos nos conjuntos de TESTE e VALIDAÇÃO...")
    
    results = []
    for model_path in model_files:
        result = evaluate_single_model(model_path)
        if result:
            results.append(result)

    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="test_accuracy", ascending=False)
        df_results.to_csv("resultados_validacao_final.csv", index=False)
        print("Validação concluída. Resultados em 'resultados_validacao_final.csv'.")
    else:
        print("Nenhum resultado gerado.")

if __name__ == "__main__":
    main()
