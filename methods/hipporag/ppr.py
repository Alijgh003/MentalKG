"""Sparse weighted PPR adapted from the retrieval flow in HippoRAG 2."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def transition_matrix(adjacency: sparse.spmatrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    matrix = sparse.csr_matrix(adjacency, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    if matrix.nnz and np.min(matrix.data) < 0:
        raise ValueError("PPR edge weights cannot be negative")
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0)
    return sparse.diags(inverse).dot(matrix).tocsr(), degrees == 0


def run_personalized_pagerank(
    transition: sparse.csr_matrix,
    reset_weights: np.ndarray,
    *,
    damping: float = 0.5,
    dangling: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> tuple[np.ndarray, int]:
    """Return PPR scores and iteration count.

    HippoRAG 2 calls igraph's weighted, undirected personalized PageRank with a
    query-specific reset vector. This equivalent power iteration keeps that
    contract while operating on the sparse PostgreSQL graph projection.
    """
    if not 0.0 < damping < 1.0:
        raise ValueError("damping must be strictly between 0 and 1")
    reset = np.asarray(reset_weights, dtype=np.float64).reshape(-1)
    if transition.shape != (reset.size, reset.size):
        raise ValueError("Reset vector size must match the transition matrix")
    reset = np.where(np.isfinite(reset) & (reset > 0), reset, 0.0)
    total = float(reset.sum())
    if total <= 0:
        raise ValueError("PPR requires at least one positive reset weight")
    reset /= total
    dangling_mask = np.asarray(dangling, dtype=bool) if dangling is not None else (
        np.asarray(transition.sum(axis=1)).ravel() == 0
    )

    scores = reset.copy()
    for iteration in range(1, max_iterations + 1):
        dangling_mass = float(scores[dangling_mask].sum())
        updated = (
            damping * np.asarray(transition.T.dot(scores)).ravel()
            + damping * dangling_mass * reset
            + (1.0 - damping) * reset
        )
        if np.linalg.norm(updated - scores, ord=1) <= tolerance:
            return updated / updated.sum(), iteration
        scores = updated
    return scores / scores.sum(), max_iterations
