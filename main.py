import argparse
from pathlib import Path

from tqdm import tqdm

from scripts.plot import run_plot
from scripts.run_bayesian_search import run_bayesian_search
from scripts.run_preprocessing import run_preprocessing
from scripts.run_validation import run_validation
from src.data.splitter import run_split


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline Coffee NIR Quality.")
    parser.add_argument("--spectra-file", type=Path, required=True, help="Arquivo de espectros brutos.")
    parser.add_argument("--quality-file", type=Path, required=True, help="Arquivo de qualidade sensorial.")
    parser.add_argument("--recipe", type=Path, required=True, help="Arquivo YAML com a recipe da busca bayesiana.")
    return parser.parse_args()


def main():
    args = parse_args()
    steps = [
        ("Divisão de Dados", lambda: run_split(args.spectra_file, args.quality_file)),
        ("Pré-processamento", run_preprocessing),
        ("Visualização de Espectros", lambda: run_plot(args.spectra_file, args.quality_file)),
        ("Busca Bayesiana", lambda: run_bayesian_search(args.recipe)),
        ("Validação Final", run_validation),
    ]
    for name, func in tqdm(steps, desc="Pipeline", unit="etapa"):
        tqdm.write(f"> {name}")
        func()


if __name__ == "__main__":
    main()
