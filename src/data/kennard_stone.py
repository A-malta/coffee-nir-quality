import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def kennard_stone(X: ArrayLike, n_samples: int) -> list[int]:
    """Seleciona amostras representativas pelo algoritmo Kennard-Stone dentro de cada classe.

    Padroniza os atributos, inicia a seleção com as amostras mais distantes e
    adiciona iterativamente a amostra com maior distância mínima em relação ao
    conjunto já selecionado.

    Args:
        X: Matriz de dados com uma amostra por linha.
        n_samples: Quantidade de amostras a selecionar.

    Returns:
        Lista de índices inteiros das amostras selecionadas.
    """
    if n_samples <= 0:
        return []

    if n_samples >= len(X):
        return list(range(len(X)))

    X = StandardScaler().fit_transform(X)
    distances = cdist(X, X)
    if n_samples == 1:
        return [int(np.argmax(distances.mean(axis=1)))]

    selected = list(map(int, np.unravel_index(np.argmax(distances), distances.shape)))

    while len(selected) < n_samples:
        remaining = [i for i in range(len(X)) if i not in selected]
        selected.append(int(remaining[np.argmax([min(distances[i, selected]) for i in remaining])]))

    return selected
