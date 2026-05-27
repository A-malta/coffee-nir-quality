import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import infer_class_target_column
from src.data.splitter import DataSplitter


RAW_SPLIT_DIR = Path("data/raw_split")
DEFAULT_SPECTRA = Path("data/RawSpectra_RoastedCoffee.xlsx")
DEFAULT_QUALITY = Path("data/SensoryQuality_RoastedCoffee.xlsx")


def load_spectra(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_excel(path, sheet_name="RawSpectra_RoastedCoffee")
    return df.iloc[:, 0], df.iloc[:, 1:]


def load_quality(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Cup quality_RoastedCoffee")
    df.columns = df.columns.str.strip()
    return df


def aligned_labels(quality: pd.DataFrame, sample_ids: list[str]) -> pd.Series:
    sample_col = quality.columns[0]
    labels = quality.set_index(sample_col)[infer_class_target_column(quality)].reindex(sample_ids)
    if labels.isna().any():
        raise ValueError(f"Amostras sem classe: {labels[labels.isna()].index.tolist()}")
    return labels


def save_spectra(wavelengths: pd.Series, spectra: pd.DataFrame, idx: list[int], name: str) -> None:
    out = pd.concat([wavelengths, spectra[[spectra.columns[i] for i in idx]]], axis=1)
    out.to_excel(RAW_SPLIT_DIR / name, index=False)


def save_quality(quality: pd.DataFrame, sample_ids: list[str], name: str) -> None:
    sample_col = quality.columns[0]
    out = pd.DataFrame({sample_col: sample_ids}).merge(quality, on=sample_col, how="left")
    if out.isna().any(axis=1).any():
        raise ValueError(f"Amostras sem qualidade em {name}")
    out.to_excel(RAW_SPLIT_DIR / name, index=False)


def run_split(spectra_file: Path = DEFAULT_SPECTRA, quality_file: Path = DEFAULT_QUALITY) -> None:
    wavelengths, spectra = load_spectra(spectra_file)
    quality = load_quality(quality_file)
    sample_ids = spectra.columns.tolist()
    labels = aligned_labels(quality, sample_ids)
    train_idx, val_idx = DataSplitter().split_train_validation(spectra.T.to_numpy(), labels.to_numpy())

    RAW_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    save_spectra(wavelengths, spectra, train_idx, "training_spectra.xlsx")
    save_spectra(wavelengths, spectra, val_idx, "validation_spectra.xlsx")
    save_quality(quality, [sample_ids[i] for i in train_idx], "training_quality.xlsx")
    save_quality(quality, [sample_ids[i] for i in val_idx], "validation_quality.xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa split treino/validação.")
    parser.add_argument("--spectra-file", type=Path, default=DEFAULT_SPECTRA)
    parser.add_argument("--quality-file", type=Path, default=DEFAULT_QUALITY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_split(args.spectra_file, args.quality_file)


if __name__ == "__main__":
    main()
