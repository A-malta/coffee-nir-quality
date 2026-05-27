import numpy as np

from .kennard_stone import kennard_stone


class DataSplitter:
    def __init__(self, validation_ratio: float = 0.2):
        self.validation_ratio = validation_ratio

    def split_train_validation(self, X: np.ndarray, y: np.ndarray) -> tuple[list[int], list[int]]:
        y = np.asarray(y)
        val_idx = []

        for label in np.unique(y):
            class_idx = np.flatnonzero(y == label)
            if len(class_idx) < 2:
                raise ValueError(f"Classe {label!r} precisa ter ao menos 2 amostras.")
            n_val = min(max(round(len(class_idx) * self.validation_ratio), 1), len(class_idx) - 1)
            val_idx.extend(class_idx[kennard_stone(X[class_idx], n_val)].tolist())

        val_idx = sorted(val_idx)
        val_set = set(val_idx)
        train_idx = [i for i in range(len(X)) if i not in val_set]
        return train_idx, val_idx
