import pandas as pd
import numpy as np
from typing import Iterable, Optional, Dict

def area_normalization(X: pd.DataFrame, wavelengths: Iterable[float]) -> pd.DataFrame:
    """Normaliza cada espectro pela área sob a curva.

    Args:
        X: DataFrame com espectros.
        wavelengths: Eixo espectral usado na integração.

    Returns:
        DataFrame normalizado por área.
    """
    wl = np.asarray(list(wavelengths), dtype=float)
    result = {}
    for col in X.columns:
        area = np.trapz(X[col], wl)
        result[col] = (X[col] / area) if area != 0 else np.zeros_like(X[col])
    return pd.DataFrame(result, index=X.index)

def msc(X: pd.DataFrame, ref: Optional[pd.Series] = None) -> pd.DataFrame:
    """Aplica Multiplicative Scatter Correction (MSC).

    Args:
        X: DataFrame com espectros.
        ref: Espectro de referência; se ``None``, usa média linha a linha.

    Returns:
        DataFrame corrigido por espalhamento multiplicativo.
    """
    ref_vec = (X.mean(axis=1) if ref is None else ref).to_numpy().reshape(-1, 1)
    A = np.hstack([np.ones_like(ref_vec), ref_vec])
    
    AtA_inv = np.linalg.pinv(A.T @ A)
    At = A.T

    corrected: Dict[str, np.ndarray] = {}
    for col in X.columns:
        y = X[col].to_numpy().reshape(-1, 1)
        coef = AtA_inv @ (At @ y)
        a, b = float(coef[0]), float(coef[1])
        corrected[str(col)] = (y[:, 0] - a) / b if b != 0 else y[:, 0] - a

    return pd.DataFrame(corrected, index=X.index)

def isc_iterative_msc(X: pd.DataFrame, iters: int = 2) -> pd.DataFrame:
    """Executa MSC iterativo atualizando referência a cada iteração.

    Args:
        X: DataFrame com espectros.
        iters: Quantidade de iterações de correção.

    Returns:
        DataFrame corrigido após iterações de MSC.
    """
    result = X.copy()
    ref = result.mean(axis=1)
    for _ in range(max(1, iters)):
        result = msc(result, ref)
        ref = result.mean(axis=1)
    return result

def emsc(X: pd.DataFrame, wavelengths: Iterable[float], ref: Optional[pd.Series] = None, poly_order: int = 2) -> pd.DataFrame:
    """Aplica Extended MSC com tendência polinomial de baseline.

    Args:
        X: DataFrame com espectros.
        wavelengths: Eixo espectral dos dados.
        ref: Espectro de referência; se ``None``, usa média linha a linha.
        poly_order: Ordem do termo polinomial do baseline.

    Returns:
        DataFrame corrigido por EMSC.
    """
    wl = np.asarray(list(wavelengths), dtype=float)
    x = (wl - wl.min()) / (wl.max() - wl.min())
    
    B = np.vstack([x**k for k in range(poly_order + 1)]).T
    
    ref_vec = (X.mean(axis=1) if ref is None else ref).to_numpy()
    
    D = np.column_stack([B, ref_vec])

    DtD_inv = np.linalg.pinv(D.T @ D)
    Dt = D.T

    corrected: Dict[str, np.ndarray] = {}
    for col in X.columns:
        y = X[col].to_numpy()
        coef = DtD_inv @ (Dt @ y)
        
        baseline = B @ coef[:B.shape[1]]
        b = coef[-1]
        
        corrected[str(col)] = (y - baseline) / b if b != 0 else y - baseline

    return pd.DataFrame(corrected, index=X.index)
