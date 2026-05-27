import argparse
from pathlib import Path

from scripts.run_grid_search import main as run_grid_search
from scripts.run_preprocessing import main as run_preprocessing
from scripts.run_split import run_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o pipeline Coffee NIR Quality.")
    parser.add_argument("--spectra-file", type=Path, help="Arquivo de espectros brutos.")
    parser.add_argument("--quality-file", type=Path, help="Arquivo de qualidade sensorial.")
    return parser.parse_args()


def split_step(args: argparse.Namespace):
    if bool(args.spectra_file) != bool(args.quality_file):
        raise ValueError("Informe --spectra-file e --quality-file juntos.")
    if args.spectra_file:
        return lambda: run_split(args.spectra_file, args.quality_file)
    return run_split


def main() -> None:
    args = parse_args()
    steps = [
        ("Divisão de Dados", split_step(args)),
        ("Pré-processamento", run_preprocessing),
        ("Grid Search", run_grid_search),
    ]

    print("=" * 50)
    print("   INICIANDO PIPELINE COFFEE-NIR-QUALITY")
    print("=" * 50)

    for name, func in steps:
        print(f"\n>>> INÍCIO: {name}")
        func()
        print(f"<<< FIM: {name}")

    print("\n" + "=" * 50)
    print("   PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 50)


if __name__ == "__main__":
    main()
