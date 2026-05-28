import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def kennard_stone(X: np.ndarray, n_samples: int) -> list[int]:
    """Seleciona amostras representativas pelo algoritmo Kennard-Stone.

    - Padroniza a escala dos dados;
    - Calcula as distâncias euclidianas entre todas as amostras;
    - Começa selecionando as duas amostras mais distantes;
    - Adiciona iterativamente a amostra mais distante do conjunto já selecionado;
    - Retorna dos indices das amostras escolhidas.

    Args:
        X: Matriz de dados com uma amostra por linha.
        n_samples: Quantidade de amostras a selecionar.

    Returns:
        Índices das amostras selecionadas.
    """
    X = StandardScaler().fit_transform(X)
    distances = cdist(X, X)
    selected = list(map(int, np.unravel_index(np.argmax(distances), distances.shape)))

    while len(selected) < n_samples:
        remaining = [i for i in range(len(X)) if i not in selected]
        selected.append(int(remaining[np.argmax([min(distances[i, selected]) for i in remaining])]))

    return selected
