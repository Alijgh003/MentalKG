"""Deterministic, thresholded consolidation within precomputed clusters."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .clustering import l2_normalize


ALGORITHM_VERSION = "direct-representative-v1"
CANONICAL_ENTITY_NAMESPACE = uuid.UUID("f8d79244-190e-4a82-8605-1dfb14273710")
CANONICAL_PREDICATE_NAMESPACE = uuid.UUID("b15ea11d-0889-4b44-8774-96c034a7b97e")


def normalize_text(value: str) -> str:
    # Must stay byte-for-byte compatible with embed_ingestion.py metadata.
    # More aggressive whitespace or punctuation normalization belongs in a
    # separately versioned consolidation rule, not in identity lookup.
    return value.strip().casefold()


@dataclass(frozen=True)
class UniqueTerm:
    text: str
    normalized_text: str
    source_record_id: str
    mention_count: int
    cluster_id: str
    vector: np.ndarray


@dataclass(frozen=True)
class ConsolidationGroup:
    representative: UniqueTerm
    members: tuple[UniqueTerm, ...]
    similarities: tuple[float, ...]


def consolidate_cluster(
    terms: list[UniqueTerm], threshold: float
) -> list[ConsolidationGroup]:
    """Greedily cover terms while requiring direct similarity to the representative."""
    if not terms:
        return []
    vectors = l2_normalize(np.asarray([term.vector for term in terms], dtype=np.float32))
    similarity = vectors @ vectors.T
    remaining = set(range(len(terms)))
    groups = []

    while remaining:
        candidates = sorted(remaining)

        def representative_key(index: int):
            covered = [other for other in candidates if similarity[index, other] > threshold]
            weighted_coverage = sum(terms[other].mention_count for other in covered)
            weighted_similarity = sum(
                float(similarity[index, other]) * terms[other].mention_count
                for other in covered
            )
            return (
                weighted_coverage,
                weighted_similarity,
                terms[index].mention_count,
                -len(terms[index].normalized_text),
                terms[index].normalized_text,
            )

        representative_index = max(candidates, key=representative_key)
        member_indices = [
            index
            for index in candidates
            if similarity[representative_index, index] > threshold
        ]
        member_indices.sort(
            key=lambda index: (
                -float(similarity[representative_index, index]),
                terms[index].normalized_text,
            )
        )
        groups.append(
            ConsolidationGroup(
                representative=terms[representative_index],
                members=tuple(terms[index] for index in member_indices),
                similarities=tuple(
                    float(similarity[representative_index, index])
                    for index in member_indices
                ),
            )
        )
        remaining.difference_update(member_indices)
    return groups


def read_cluster_manifest(path: Path) -> dict[str, list[dict]]:
    clusters = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                clusters[record["cluster_id"]].append(record)
    return dict(clusters)


def manifest_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_id(kind: str, entity_type: str, normalized_name: str) -> uuid.UUID:
    namespace = (
        CANONICAL_ENTITY_NAMESPACE if kind == "entity" else CANONICAL_PREDICATE_NAMESPACE
    )
    return uuid.uuid5(namespace, f"{entity_type}:{normalized_name}")
