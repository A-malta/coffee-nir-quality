import numpy as np
from typing import Tuple, List
from .kennard_stone import kennard_stone

class DataSplitter:
    def __init__(self, test_val_ratio: float = 0.2, val_ratio_relative: float = 0.5):
        self.test_val_ratio = test_val_ratio
        self.val_ratio_relative = val_ratio_relative

    def split_indices(self, X: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        n_total = X.shape[0]
        n_test_val = int(self.test_val_ratio * n_total)
        
        idx_test_val = kennard_stone(X, n_test_val)
        
        all_indices = set(range(n_total))
        idx_train = list(all_indices - set(idx_test_val))
        
        X_test_val = X[idx_test_val]
        n_val = int(self.val_ratio_relative * len(idx_test_val))
        
        idx_val_relative = kennard_stone(X_test_val, n_val)
        
        idx_val = [idx_test_val[i] for i in idx_val_relative]
        idx_test = list(set(idx_test_val) - set(idx_val))
        
        return idx_train, idx_test, idx_val
