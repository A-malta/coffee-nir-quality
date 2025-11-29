import sys
import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.splitter import DataSplitter

def load_spectra(file_path: Path) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    print(f"Lendo espectros de: {file_path}")
    df_spectra = pd.read_excel(file_path, sheet_name="RawSpectra_RoastedCoffee")
    wavelengths = df_spectra.iloc[:, 0]
    spectra = df_spectra.iloc[:, 1:]
    return wavelengths, spectra

def save_split_data(wavelengths: pd.Series, spectra: pd.DataFrame, indices: List[int], filename: str):
    cols = [spectra.columns[i] for i in indices]
    data = pd.concat([wavelengths, spectra[cols]], axis=1)
    data.to_excel(f"data/raw_split/{filename}", index=False)
    print(f"Salvo: data/raw_split/{filename}")

def process_spectra_split(file_path: Path) -> Tuple[List[str], List[str], List[str]]:
    wavelengths, spectra = load_spectra(file_path)
    X = spectra.T.values
    
    splitter = DataSplitter()
    idx_train, idx_test, idx_val = splitter.split_indices(X)
    
    print(f"Split realizado -> Treino: {len(idx_train)}, Teste: {len(idx_test)}, Validação: {len(idx_val)}")
    
    os.makedirs("data/raw_split", exist_ok=True)
    save_split_data(wavelengths, spectra, idx_train, "training_spectra.xlsx")
    save_split_data(wavelengths, spectra, idx_test, "test_spectra.xlsx")
    save_split_data(wavelengths, spectra, idx_val, "validation_spectra.xlsx")
    
    all_cols = spectra.columns.tolist()
    return [all_cols[i] for i in idx_train], [all_cols[i] for i in idx_test], [all_cols[i] for i in idx_val]

def filter_and_save_quality(df_quality: pd.DataFrame, cols: List[str], filename: str, code_col: str):
    subset = df_quality[df_quality[code_col].isin(cols)]
    subset.to_excel(f"data/raw_split/{filename}", index=False)

def process_quality_split(file_path: Path, train_cols: List[str], test_cols: List[str], val_cols: List[str]):
    print(f"Lendo qualidade de: {file_path}")
    df_quality = pd.read_excel(file_path, sheet_name="Cup quality_RoastedCoffee")
    df_quality.columns = df_quality.columns.str.strip() 
    code_col = df_quality.columns[0]
    
    filter_and_save_quality(df_quality, train_cols, "training_quality.xlsx", code_col)
    filter_and_save_quality(df_quality, test_cols, "test_quality.xlsx", code_col)
    filter_and_save_quality(df_quality, val_cols, "validation_quality.xlsx", code_col)

def main():
    base_path = Path("data/raw")
    spectra_file = base_path / "RawSpectra_RoastedCoffee.xlsx"
    quality_file = base_path / "SensoryQuality_RoastedCoffee.xlsx"
    
    if not spectra_file.exists():
        print(f"Erro: Arquivo não encontrado: {spectra_file}")
        print(f"Por favor, coloque os arquivos originais em '{base_path.absolute()}'")
        return

    print("Iniciando processamento Kennard-Stone 80/10/10...")
    train_cols, test_cols, val_cols = process_spectra_split(spectra_file)
    process_quality_split(quality_file, train_cols, test_cols, val_cols)
    print("Processamento concluído.")

if __name__ == "__main__":
    main()
