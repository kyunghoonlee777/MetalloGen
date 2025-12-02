import numpy as np
from MetalloGen import globalvars as gv

def _center(X):
    """Center the points in X to their centroid."""
    c = X.mean(axis=0)
    c = X[0]
    return X - c, c 

def kabsch_align_rmsd(P, Q):
    """The Kabsch algorithm that computes the optimal rotation matrix
    that minimizes the RMSD between two sets of points P and Q.
    
    Args:
        P (np.ndarray): An Nx3 matrix of points.
        Q (np.ndarray): An Nx3 matrix of points.
    
    Returns:
        rmsd (float): The root mean square deviation between P and Q after alignment.
    """
    # Center the points
    P_centered, P_centroid = _center(P)
    Q_centered, Q_centroid = _center(Q)
    
    # Compute covariance matrix
    C = P_centered.T @ Q_centered

    # Singular Value Decomposition
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1.0, 1.0, d]) @ Wt

    diff = P_centered @ U - Q_centered
    mse = np.mean(np.sum(diff**2, axis=1))

    rmsd = np.sqrt(mse)
    return rmsd

def build_mask(atom_symbols, exclude_H=True):
    if not exclude_H:
        return np.ones(len(atom_symbols), dtype=bool)
    else:
        return np.array([s != 'H' for s in atom_symbols], dtype=bool)

def pairwise_aligned_rmsd_matrix(conformers, atom_symbols=None, exclude_H=True):
    n = len(conformers)
    if n == 0:
        return np.zeros((0, 0))
    na = conformers[0].shape[0]

    mask = build_mask(atom_symbols, exclude_H)
    idx = np.where(mask)[0]
    Xs = [conf[idx] for conf in conformers]
    
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        Pi = Xs[i]
        for j in range(i + 1, n):
            Qj = Xs[j]
            rmsd = kabsch_align_rmsd(Pi, Qj)
            D[i, j] = rmsd
            D[j, i] = rmsd
    return D

def butina_clusters_from_distance_matrix(D, cutoff):
    """
    Perform Butina clustering on a distance matrix.
    - D: Distance matrix (2D numpy array).
    - cutoff: Distance cutoff for clustering.
    
    Returns:
        List of clusters, where each cluster is a list of indices.
    """
    n = D.shape[0]
    if n == 0:
        return []
    neighbors = {i: set(np.where(D[i] < cutoff)[0]) - {i} for i in range(n)}
    unused = set(range(n))
    clusters = []
    while unused:
        best = max(unused, key=lambda i: len(neighbors[i] & unused))
        cl = [best] + list(neighbors[best] & unused)
        cl = sorted(cl)
        for idx in cl:
            unused.discard(idx)
        clusters.append(cl)
    clusters.sort(key=len, reverse=True)
    return clusters

def cluster_conformers_butina(conformers, atom_symbol=None, cutoff=0.75, exclude_H=True):
    D = pairwise_aligned_rmsd_matrix(conformers, atom_symbol, exclude_H)
    clusters = butina_clusters_from_distance_matrix(D, cutoff)
    return clusters, D
    