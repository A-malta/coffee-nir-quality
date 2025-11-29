import pandas as pd
import numpy as np
from typing import Dict

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
except ImportError:
    sparse = None
    spsolve = None

def baseline_asls(X: pd.DataFrame, lam: float, p: float, niter: int) -> pd.DataFrame:
    if sparse is None or spsolve is None:
        raise ImportError("A biblioteca 'scipy' é necessária para esta função.")

    n = X.shape[0]
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2))
    H = lam * (D @ D.T)

    result: Dict[str, np.ndarray] = {}
    for col in X.columns:
        y = X[col].to_numpy()
        w = np.ones(n)
        z = np.zeros(n) 
        for _ in range(niter):
            W = sparse.diags(w, 0)
            Z = W + H
            z = spsolve(Z.tocsc(), w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        result[str(col)] = y - z

    return pd.DataFrame(result, index=X.index)
