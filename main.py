import sys
import os
from typing import Callable, List, Tuple
from scripts.run_split import main as run_split
from scripts.run_preprocessing import main as run_preprocessing
from scripts.run_grid_search import main as run_grid_search
from scripts.run_validation import main as run_validation

def run_step(step_name: str, step_func: Callable[[], None]) -> bool:
    """Executa uma etapa do pipeline e retorna se ela concluiu com sucesso.

    Args:
        step_name: Nome descritivo da etapa para logs.
        step_func: Função sem argumentos que executa a etapa.

    Returns:
        True quando a etapa termina sem exceção; caso contrário, False.
    """
    print(f"\n>>> ETAPA: {step_name}")
    try:
        step_func()
        return True
    except Exception as e:
        print(f"Erro na etapa '{step_name}': {e}")
        return False

def main():
    """Executa o fluxo principal do script.

    Returns:
        None.
    """
    print("=" * 50)
    print("   INICIANDO PIPELINE COFFEE-NIR-QUALITY")
    print("=" * 50)

    steps: List[Tuple[str, Callable[[], None]]] = [
        ("Divisão de Dados (Split)", run_split),
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
