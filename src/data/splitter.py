import numpy as np
from typing import Tuple, List
from .kennard_stone import kennard_stone

class DataSplitter:
    def __init__(self, test_val_ratio: float = 0.2, val_ratio_relative: float = 0.5):
        self.test_val_ratio = test_val_ratio
        self.val_ratio_relative = val_ratio_relative

    def split_indices(self, X: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        n_total = X.shape[0]
        

        n_train = int((1 - self.test_val_ratio) * n_total)
        idx_train = kennard_stone(X, n_train)
        

        all_indices = set(range(n_total))
        remaining_indices = list(all_indices - set(idx_train))
        X_remaining = X[remaining_indices]
        

        n_test = int(len(remaining_indices) * 0.5)
        idx_test_relative = kennard_stone(X_remaining, n_test)
        idx_test = [remaining_indices[i] for i in idx_test_relative]
        

        idx_val = list(set(remaining_indices) - set(idx_test))
        
        return idx_train, idx_test, idx_val
