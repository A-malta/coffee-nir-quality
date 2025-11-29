import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from typing import List

def kennard_stone(X: np.ndarray, n_samples: int) -> List[int]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dist_matrix = cdist(X_scaled, X_scaled)
    
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    selected = [int(i), int(j)]
    
    while len(selected) < n_samples:
        remaining = list(set(range(len(X_scaled))) - set(selected))
        
        min_distances = [min(dist_matrix[k, selected]) for k in remaining]
        
        next_point = remaining[np.argmax(min_distances)]
        selected.append(int(next_point))
        
    return selected
