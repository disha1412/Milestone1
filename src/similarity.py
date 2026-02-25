"""
Vectorized cosine similarity and Euclidean distance for batches of vectors.
Both functions operate on (N, D) NumPy arrays and return length-N arrays.
"""

import numpy as np


def cosine_similarity_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity for each pair of rows in a and b.
    a, b: (N, D) arrays
    Returns: (N,) array of cosine similarities in [-1, 1]
    """
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    denom = norm_a * norm_b
    # Avoid division by zero
    denom = np.where(denom == 0, 1e-10, denom)
    return dot / denom


def euclidean_distance_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance for each pair of rows in a and b.
    a, b: (N, D) arrays
    Returns: (N,) array of distances >= 0
    """
    return np.linalg.norm(a - b, axis=1)


def cosine_similarity_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Baseline: Python loop cosine similarity. Slow — for benchmarking only."""
    n = a.shape[0]
    out = np.empty(n)
    for i in range(n):
        dot = np.dot(a[i], b[i])
        denom = np.linalg.norm(a[i]) * np.linalg.norm(b[i])
        out[i] = dot / denom if denom > 1e-10 else 0.0
    return out


def euclidean_distance_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Baseline: Python loop Euclidean distance. Slow — for benchmarking only."""
    n = a.shape[0]
    out = np.empty(n)
    for i in range(n):
        out[i] = np.linalg.norm(a[i] - b[i])
    return out