import numpy as np


def pairwise_similarity_matrix(vectors, kind):
    """vectors: array (H, D). kind: 'cosine' o 'pearson'. Devuelve (H, H) simétrica, diagonal 1."""
    H = vectors.shape[0]
    M = np.eye(H)
    for i in range(H):
        for j in range(i + 1, H):
            a, b = vectors[i], vectors[j]
            if kind == "cosine":
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                s = float(a @ b / denom) if denom > 0 else np.nan
            elif kind == "pearson":
                if np.std(a) == 0 or np.std(b) == 0:
                    s = np.nan
                else:
                    s = float(np.corrcoef(a, b)[0, 1])
            else:
                raise ValueError(kind)
            M[i, j] = M[j, i] = s
    return M


def offdiag_mean(M):
    """Promedio de las entradas fuera de la diagonal (triángulo superior)."""
    iu = np.triu_indices(M.shape[0], k=1)
    return float(np.nanmean(M[iu]))
