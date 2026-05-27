import os
from typing import Optional, Tuple

import pandas as pd


CLASS_TARGET_CANDIDATES = ("Class", "label", "Label", "Classe")


def load_excel_data(file_path: str, sheet_name: Optional[str] = 0, header: Optional[int] = 0) -> pd.DataFrame:
    """Load an Excel sheet into a DataFrame.

    Args:
        file_path: Excel file path.
        sheet_name: Sheet name or index.
        header: Header row index.

    Returns:
        Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    return pd.read_excel(file_path, sheet_name=sheet_name, header=header)


def infer_class_target_column(y_df: pd.DataFrame) -> str:
    """Infer the categorical class target column from a quality table.

    The sensory score column is numeric, so known class labels and categorical
    columns are preferred to avoid training a classifier on a continuous target.
    """
    y_df.columns = y_df.columns.astype(str).str.strip()
    normalized_columns = {column.lower(): column for column in y_df.columns}

    for candidate in CLASS_TARGET_CANDIDATES:
        column = normalized_columns.get(candidate.lower())
        if column is not None:
            return column

    categorical_columns = [
        column
        for column in y_df.columns
        if (
            pd.api.types.is_object_dtype(y_df[column])
            or pd.api.types.is_bool_dtype(y_df[column])
        )
    ]

    if len(categorical_columns) == 1:
        return categorical_columns[0]

    raise KeyError(
        "Não foi possível identificar a coluna de classe. "
        "Use uma coluna chamada 'Class' ou 'label'. "
        f"Colunas encontradas: {list(y_df.columns)}"
    )


def split_features_target(X_df: pd.DataFrame, y_df: pd.DataFrame, target_col: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
    """Align features and target by position.

    Args:
        X_df: Feature table.
        y_df: Target table.
        target_col: Target column name in ``y_df``.

    Returns:
        A tuple ``(X, y)`` with matching lengths.
    """
    if target_col not in y_df.columns:
        raise ValueError(f"Coluna '{target_col}' não encontrada no DataFrame de target.")

    min_len = min(len(X_df), len(y_df))
    X = X_df.iloc[:min_len]
    y = y_df.iloc[:min_len][target_col]

    return X, y
