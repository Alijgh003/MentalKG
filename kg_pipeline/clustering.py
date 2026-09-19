"""Type-aware candidate clustering for unconsolidated KG vectors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


SEED_ELIGIBLE_ENTITY_TYPES = ("symptom", "behavior", "disorder", "concept")


@dataclass(frozen=True)
class ClusterPolicy:
    """Controls candidate-bucket granularity, not a clinical class count."""

    sqrt_factor: float
    minimum_clusters: int = 2


ENTITY_CLUSTER_POLICIES = {
    # Symptom and concept vocabularies are heterogeneous, so keep their
    # candidate buckets relatively narrow. Disorder aliases are more lexical;
    # behavior has fewer unique values and needs fewer buckets.
    "symptom": ClusterPolicy(sqrt_factor=3.0),
    "behavior": ClusterPolicy(sqrt_factor=2.0),
    "disorder": ClusterPolicy(sqrt_factor=3.5),
    "concept": ClusterPolicy(sqrt_factor=3.0),
}
PREDICATE_CLUSTER_POLICY = ClusterPolicy(sqrt_factor=3.0)


def choose_cluster_count(unique_count: int, policy: ClusterPolicy) -> int:
    """Choose a conservative candidate-cluster count from vocabulary size."""
    if unique_count <= 1:
        return unique_count
    proposed = round(policy.sqrt_factor * math.sqrt(unique_count))
    return min(unique_count, max(policy.minimum_clusters, proposed))


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize embeddings so Euclidean K-Means behaves like cosine grouping."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot cluster zero-length embedding vectors")
    return matrix / norms


def cluster_vectors(
    matrix: np.ndarray,
    cluster_count: int,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, normalized centroids, and cosine similarity to centroid."""
    from sklearn.cluster import MiniBatchKMeans

    normalized = l2_normalize(matrix)
    if cluster_count == 1:
        labels = np.zeros(len(normalized), dtype=np.int32)
        centroid = l2_normalize(normalized.mean(axis=0, keepdims=True))
        similarities = normalized @ centroid[0]
        return labels, centroid, similarities

    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=min(2048, max(256, cluster_count * 4)),
        n_init=3,
        max_iter=200,
        reassignment_ratio=0.01,
        random_state=random_state,
        verbose=0,
    )
    labels = model.fit_predict(normalized)
    centroids = l2_normalize(model.cluster_centers_)
    similarities = np.einsum("ij,ij->i", normalized, centroids[labels])
    return labels, centroids, similarities
