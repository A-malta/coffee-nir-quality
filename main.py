import argparse
from typing import Callable, List, Tuple

from scripts.run_grid_search import main as run_grid_search
from scripts.run_preprocessing import main as run_preprocessing
from scripts.run_split import main as run_split
from scripts.run_split import process_quality_split, process_spectra_split
from scripts.run_validation import main as run_validation


def run_step(step_name: str, step_func: Callable[[], None]) -> bool:
    """Executa uma etapa do pipeline e retorna se ela concluiu com sucesso."""
    print(f"\n>>> ETAPA: {step_name}")
    try:
        step_func()
        return True
    except Exception as error:
        print(f"Erro na etapa '{step_name}': {error}")
        return False


def parse_args() -> argparse.Namespace:
    """Define e lê argumentos CLI do pipeline principal."""
    parser = argparse.ArgumentParser(description="Executa o pipeline Coffee NIR Quality.")
    parser.add_argument(
        "--spectra-file",
        type=str,
        default=None,
        help="Caminho do arquivo de espectros brutos (opcional).",
    )
    parser.add_argument(
        "--quality-file",
        type=str,
        default=None,
        help="Caminho do arquivo de qualidade sensorial (opcional).",
    )
    return parser.parse_args()


def run_split_with_custom_inputs(spectra_file: str, quality_file: str) -> None:
    """Executa o split usando arquivos de entrada informados via CLI."""
    train_cols, test_cols, val_cols = process_spectra_split(spectra_file)
    process_quality_split(quality_file, train_cols, test_cols, val_cols)


def main() -> None:
    """Executa o fluxo principal do pipeline."""
    args = parse_args()

    print("=" * 50)
    print("   INICIANDO PIPELINE COFFEE-NIR-QUALITY")
    print("=" * 50)

    split_step: Callable[[], None]
    if args.spectra_file and args.quality_file:
        split_step = lambda: run_split_with_custom_inputs(args.spectra_file, args.quality_file)
    elif args.spectra_file or args.quality_file:
        raise ValueError("Para usar arquivos customizados, informe --spectra-file e --quality-file juntos.")
    else:
        split_step = run_split

    steps: List[Tuple[str, Callable[[], None]]] = [
        ("Divisão de Dados (Split)", split_step),
        ("Pré-processamento", run_preprocessing),
        ("Grid Search (Treinamento e Otimização)", run_grid_search),
        ("Validação Final", run_validation),
    ]

    for name, func in steps:
        if not run_step(name, func):
            print("\nPipeline interrompido devido a erro.")
            return

    print("\n" + "=" * 50)
    print("   PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 50)


if __name__ == "__main__":
    main()
