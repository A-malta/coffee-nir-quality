import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def kennard_stone(X: np.ndarray, n_samples: int) -> list[int]:
    if n_samples <= 0:
        return []
    if n_samples >= len(X):
        return list(range(len(X)))

    X = StandardScaler().fit_transform(X)
    if n_samples == 1:
        return [int(np.argmin(cdist(X, X.mean(axis=0, keepdims=True)).ravel()))]

    distances = cdist(X, X)
    selected = list(map(int, np.unravel_index(np.argmax(distances), distances.shape)))

    while len(selected) < n_samples:
        remaining = [i for i in range(len(X)) if i not in selected]
        selected.append(int(remaining[np.argmax([min(distances[i, selected]) for i in remaining])]))

    return selected
