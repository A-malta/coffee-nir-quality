import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.filters import mean_centering, savitzky_derivative, sg_smoothing

BASE_OUT_DIR = "data/processed"
OUTPUT_FILENAME = "SG_Smoothing+1D+2D+MeanCentering.xlsx"


def read_raw_spectra_excel(file_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_excel(file_path)
    wavelengths = df.iloc[:, 0].values
    spectra = df.iloc[:, 1:]
    return wavelengths, spectra


def save_processed_spectra(wl: np.ndarray, result: pd.DataFrame, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    pd.concat([pd.Series(wl, name="Wavenumbers"), result], axis=1).to_excel(out_path, index=False)
    print(f"  -> {out_path}")


def apply_requested_pipeline(X: pd.DataFrame, wavelengths: np.ndarray) -> pd.DataFrame:
    X_processed = sg_smoothing(X, window_length=15, polyorder=2)
    X_processed = savitzky_derivative(
        X_processed,
        window_length=15,
        polyorder=2,
        deriv=1,
        wavelengths=wavelengths,
    )
    X_processed = savitzky_derivative(
        X_processed,
        window_length=15,
        polyorder=2,
        deriv=2,
        wavelengths=wavelengths,
    )
    return X_processed


def process_datasets() -> None:
    raw_dir = Path("data/raw_split")
    if not raw_dir.exists():
        print("Diretório data/raw_split não encontrado.")
        return

    try:
        wl, X_train = read_raw_spectra_excel(raw_dir / "training_spectra.xlsx")
        _, X_val = read_raw_spectra_excel(raw_dir / "validation_spectra.xlsx")
        _, X_test = read_raw_spectra_excel(raw_dir / "test_spectra.xlsx")
    except FileNotFoundError as exc:
        print(f"Erro ao carregar arquivos: {exc}")
        return


    print("Aplicando pipeline: Savitzky-Golay (alisamento) -> 1ª derivada -> 2ª derivada -> Mean Centering")

    train_after_baseline = apply_requested_pipeline(X_train, wl)
    val_after_baseline = apply_requested_pipeline(X_val, wl)
    test_after_baseline = apply_requested_pipeline(X_test, wl)

    train_mean = train_after_baseline.mean(axis=1)

    save_processed_spectra(wl, mean_centering(train_after_baseline, mean_vector=train_mean), Path(BASE_OUT_DIR) / "training", OUTPUT_FILENAME)
    save_processed_spectra(wl, mean_centering(val_after_baseline, mean_vector=train_mean), Path(BASE_OUT_DIR) / "validation", OUTPUT_FILENAME)
    save_processed_spectra(wl, mean_centering(test_after_baseline, mean_vector=train_mean), Path(BASE_OUT_DIR) / "test", OUTPUT_FILENAME)

    print("\nTotal de arquivos gerados: 3 (1 por dataset)")


def main() -> None:
    process_datasets()


if __name__ == "__main__":
    main()
