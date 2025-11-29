import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional

def load_excel_data(file_path: str, sheet_name: Optional[str] = 0, header: Optional[int] = 0) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    return pd.read_excel(file_path, sheet_name=sheet_name, header=header)

def split_features_target(X_df: pd.DataFrame, y_df: pd.DataFrame, target_col: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in y_df.columns:
        raise ValueError(f"Coluna '{target_col}' não encontrada no DataFrame de target.")
    
    min_len = min(len(X_df), len(y_df))
    X = X_df.iloc[:min_len]
    y = y_df.iloc[:min_len][target_col]
    
    return X, y
