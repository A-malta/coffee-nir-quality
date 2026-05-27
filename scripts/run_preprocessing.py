import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.filters import mean_centering, savitzky_derivative, sg_smoothing


RAW_DIR = Path("data/raw_split")
OUT_DIR = Path("data/processed")
OUT_FILE = "SG_Smoothing+1D+2D+MeanCentering.xlsx"


def read_spectra(path: Path):
    df = pd.read_excel(path)
    return df.iloc[:, 0].to_numpy(), df.iloc[:, 1:]


def preprocess(X: pd.DataFrame, wavelengths):
    X = sg_smoothing(X, window_length=15, polyorder=2)
    X = savitzky_derivative(X, window_length=15, polyorder=2, deriv=1, wavelengths=wavelengths)
    return savitzky_derivative(X, window_length=15, polyorder=2, deriv=2, wavelengths=wavelengths)


def save(wavelengths, X: pd.DataFrame, split: str) -> None:
    path = OUT_DIR / split / OUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.Series(wavelengths, name="Wavenumbers"), X], axis=1).to_excel(path, index=False)


def main() -> None:
    wavelengths, X_train = read_spectra(RAW_DIR / "training_spectra.xlsx")
    _, X_val = read_spectra(RAW_DIR / "validation_spectra.xlsx")

    X_train = preprocess(X_train, wavelengths)
    X_val = preprocess(X_val, wavelengths)
    train_mean = X_train.mean(axis=1)

    save(wavelengths, mean_centering(X_train, train_mean), "training")
    save(wavelengths, mean_centering(X_val, train_mean), "validation")


if __name__ == "__main__":
    main()
