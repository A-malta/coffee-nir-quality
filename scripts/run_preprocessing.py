import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.filters import savitzky_derivative, snv, mean_centering
from src.preprocessing.normalization import msc, area_normalization
from src.preprocessing.baseline import baseline_asls

BASE_OUT_DIR = "data/processed"
ASLS_PARAMS = {"lam": 1e5, "p": 0.01, "niter": 10}

PREPROCESSORS = {
    "1D_SavGolay": lambda X, wl: savitzky_derivative(X, 15, 2, deriv=1, wavelengths=wl),
    "2D_SavGolay": lambda X, wl: savitzky_derivative(X, 15, 2, deriv=2, wavelengths=wl),
    "MSC": lambda X, wl: msc(X),
    "SNV": lambda X, wl: snv(X),
    "MC": lambda X, wl: mean_centering(X),
    "Baseline": lambda X, wl: baseline_asls(X, **ASLS_PARAMS),
    "AreaNorm": area_normalization,
}

def read_raw_spectra_excel(file_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_excel(file_path)
    wavelengths = df.iloc[:, 0].values
    spectra = df.iloc[:, 1:]
    return wavelengths, spectra

def apply_single_step(X: pd.DataFrame, wl: np.ndarray, step_name: str) -> pd.DataFrame:
    func = PREPROCESSORS[step_name]
    return func(X, wl)

def save_processed_spectra(wl: np.ndarray, result: pd.DataFrame, output_dir: str, filename: str):
    out_path = os.path.join(output_dir, filename)
    pd.concat([pd.Series(wl, name="Wavenumbers"), result], axis=1).to_excel(out_path, index=False)
    print(f"  -> {filename}")

def process_and_save_steps(X: pd.DataFrame, wl: np.ndarray, output_dir: str):
    counter = 1
    for step_name in PREPROCESSORS:
        try:
            result = apply_single_step(X, wl, step_name)
            filename = f"{counter:03d}_{step_name}.xlsx"
            save_processed_spectra(wl, result, output_dir, filename)
            counter += 1
        except Exception as e:
            print(f"  [Erro] Falha ao aplicar {step_name}: {e}")

def process_file(file_path: str):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    dataset_type = base_name.split('_')[0] 
    
    output_dir = os.path.join(BASE_OUT_DIR, dataset_type)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processando '{base_name}'...")
    wl, X = read_raw_spectra_excel(file_path)
    
    process_and_save_steps(X, wl, output_dir)

def main():
    input_dir = Path("data/raw_split")
    if not input_dir.exists():
        print(f"Diretório {input_dir} não encontrado. Execute run_split.py primeiro.")
        return

    files = list(input_dir.glob("*_spectra.xlsx"))
    if not files:
        print("Nenhum arquivo de espectro encontrado em data/raw_split.")
        return

    for file_path in files:
        process_file(str(file_path))

if __name__ == "__main__":
    main()
