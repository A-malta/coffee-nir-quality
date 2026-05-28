import argparse
from pathlib import Path

from tqdm import tqdm

from scripts.plot import main as run_plot
from scripts.run_grid_search import main as run_grid_search
from scripts.run_preprocessing import main as run_preprocessing
from scripts.run_validation import main as run_validation
from src.data.splitter import run_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline Coffee NIR Quality.")
    parser.add_argument("--spectra-file", type=Path, required=True, help="Arquivo de espectros brutos.")
    parser.add_argument("--quality-file", type=Path, required=True, help="Arquivo de qualidade sensorial.")
    parser.add_argument("--recipe", type=Path, required=True, help="Arquivo YAML com a recipe do grid search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = [
        ("Divisão de Dados", lambda: run_split(args.spectra_file, args.quality_file)),
        ("Pré-processamento", run_preprocessing),
        ("Visualização de Espectros", lambda: run_plot(args.spectra_file, args.quality_file)),
        ("Grid Search", lambda: run_grid_search(args.recipe)),
        ("Validação Final", run_validation),
    ]
    for name, func in tqdm(steps, desc="Pipeline", unit="etapa"):
        tqdm.write(f"> {name}")
        func()


if __name__ == "__main__":
    main()
