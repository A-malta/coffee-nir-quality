import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.filters import savitzky_derivative, snv, mean_centering, moving_average, sg_smoothing
from src.preprocessing.normalization import msc, area_normalization

BASE_OUT_DIR = "data/processed"

def read_raw_spectra_excel(file_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_excel(file_path)
    wavelengths = df.iloc[:, 0].values
    spectra = df.iloc[:, 1:]
    return wavelengths, spectra

def save_processed_spectra(wl: np.ndarray, result: pd.DataFrame, output_dir: str, filename: str):
    out_path = os.path.join(output_dir, filename)
    pd.concat([pd.Series(wl, name="Wavenumbers"), result], axis=1).to_excel(out_path, index=False)
    print(f"  -> {filename}")

def process_datasets():

    raw_dir = Path("data/raw_split")
    if not raw_dir.exists():
        print("Diretório data/raw_split não encontrado.")
        return

    try:
        wl, X_train = read_raw_spectra_excel(raw_dir / "training_spectra.xlsx")
        _, X_val = read_raw_spectra_excel(raw_dir / "validation_spectra.xlsx")
        _, X_test = read_raw_spectra_excel(raw_dir / "test_spectra.xlsx")
    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos: {e}")
        return

    datasets = {"training": X_train, "validation": X_val, "test": X_test}
    

    for dtype in datasets:
        os.makedirs(os.path.join(BASE_OUT_DIR, dtype), exist_ok=True)


    group1_methods = {
        "Smoothing_SG0": lambda X, wl, ref: sg_smoothing(X, window_length=15, polyorder=0),
        "MovingAvg": lambda X, wl, ref: moving_average(X, window_length=15),
        "SG_1D": lambda X, wl, ref: savitzky_derivative(X, window_length=15, polyorder=2, deriv=1, wavelengths=wl),
        "SG_2D": lambda X, wl, ref: savitzky_derivative(X, window_length=15, polyorder=2, deriv=2, wavelengths=wl),
        "MSC": lambda X, wl, ref: msc(X, ref=ref),
        "SNV": lambda X, wl, ref: snv(X),
    }


    group2_methods = {
        "MeanCentering": "MC",
        "VarianceScaling": "VS",
        "Autoscaling": "AS"
    }

    count = 0
    

    for g1_name, g1_func in group1_methods.items():
        print(f"\nAplicando Grupo 1: {g1_name}")
        

        msc_ref = None
        if g1_name == "MSC":
            msc_ref = X_train.mean(axis=1)


        X_g1 = {}
        for dtype, X in datasets.items():
            try:
                X_g1[dtype] = g1_func(X, wl, msc_ref)
            except Exception as e:
                print(f"  [Erro] Falha G1 {g1_name} em {dtype}: {e}")
                X_g1[dtype] = None

        if any(x is None for x in X_g1.values()):
            continue


        for g2_name, g2_type in group2_methods.items():
            try:

                X_train_g1 = X_g1["training"]
                
                mu = None
                std = None
                

                if g2_type == "MC":
                    mu = X_train_g1.mean(axis=1)
                elif g2_type == "VS":
                    std = X_train_g1.std(axis=1)
                elif g2_type == "AS":
                    mu = X_train_g1.mean(axis=1)
                    std = X_train_g1.std(axis=1)


                for dtype, X_curr_g1 in X_g1.items():
                    X_final = X_curr_g1.copy()
                    

                    if g2_type == "MC":
                        X_final = mean_centering(X_final, mean_vector=mu)
                    elif g2_type == "VS":
                        X_final = X_final.div(std, axis=0).fillna(0.0)
                    elif g2_type == "AS":
                        X_final = X_final.sub(mu, axis=0).div(std, axis=0).fillna(0.0)
                    
                    filename = f"{g1_name}+{g2_name}.xlsx"
                    save_processed_spectra(wl, X_final, os.path.join(BASE_OUT_DIR, dtype), filename)
                    count += 1
                    
            except Exception as e:
                print(f"  [Erro] Falha G2 {g2_name} sobre {g1_name}: {e}")

    print(f"\nTotal de arquivos gerados: {count} (x3 datasets)")

def main():
    process_datasets()

if __name__ == "__main__":
    main()
