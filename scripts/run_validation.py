import sys
import os
import pandas as pd
import glob
from typing import Tuple, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_excel_data
from src.models.random_forest import RandomForestModel
from src.evaluation.metrics import evaluate_model

def load_dataset(type_data: str, preprocess_step: str = "007_AreaNorm") -> Tuple[pd.DataFrame, pd.Series]:
    x_path = f"data/processed/{type_data}/{preprocess_step}.xlsx"
    y_path = f"data/raw_split/{type_data}_quality.xlsx"
    
    X_raw = load_excel_data(x_path)
    y_raw = load_excel_data(y_path)
    
    X = X_raw.iloc[:, 1:].T
    
    target_col = 'label'
    if target_col not in y_raw.columns:
        if len(y_raw.columns) > 1:
            target_col = y_raw.columns[1]
            print(f"Aviso: Coluna 'label' não encontrada. Usando '{target_col}' como target.")
        else:
            raise KeyError("Não foi possível identificar a coluna de target (label).")

    y = y_raw[target_col]
    
    min_size = min(X.shape[0], len(y))
    return X.iloc[:min_size], y.iloc[:min_size]

def evaluate_single_model(model_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> Optional[Dict[str, Any]]:
    try:
        model = RandomForestModel.load(model_path)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        
        return {
            "model": os.path.basename(model_path),
            **metrics
        }
    except Exception as e:
        print(f"Erro ao avaliar {model_path}: {e}")
        return None

def main():
    preprocess_step = "007_AreaNorm"
    models_dir = "models"
    
    try:
        X_test, y_test = load_dataset("test", preprocess_step)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return

    model_files = glob.glob(os.path.join(models_dir, "*.joblib"))
    if not model_files:
        print(f"Nenhum modelo encontrado em {models_dir}")
        return

    print(f"Avaliando {len(model_files)} modelos no conjunto de TESTE...")
    
    results = []
    for model_path in model_files:
        result = evaluate_single_model(model_path, X_test, y_test)
        if result:
            results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv("resultados_validacao_final.csv", index=False)
    print("Validação concluída. Resultados em 'resultados_validacao_final.csv'.")

if __name__ == "__main__":
    main()
