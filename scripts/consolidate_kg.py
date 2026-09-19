#!/usr/bin/env python3
"""Consolidate clustered entity and predicate vectors into PostgreSQL."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psycopg

from config.settings import DatabaseSettings
from kg_pipeline.clustering import SEED_ELIGIBLE_ENTITY_TYPES
from kg_pipeline.consolidation import (
    ALGORITHM_VERSION,
    UniqueTerm,
    canonical_id,
    consolidate_cluster,
    manifest_sha256,
    normalize_text,
    read_cluster_manifest,
)
from kg_pipeline.schema import create_schema
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig

logger = logging.getLogger(__name__)


def _parser():
    parser = argparse.ArgumentParser(
        description="Consolidate within type-specific clusters using direct cosine similarity"
    )
    parser.add_argument("--cluster-dir", type=Path, default=Path("outputs/clustering"))
    parser.add_argument("--threshold", type=float, default=0.87)
    parser.add_argument("--apply", action="store_true", help="Write canonical mappings to PostgreSQL")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _fetch_vectors(store, collection: str, ids: list[str]) -> dict[str, np.ndarray]:
    vectors = {}
    for start in range(0, len(ids), 1000):
        batch = ids[start : start + 1000]
        for record in store.client.get(
            collection_name=collection,
            ids=batch,
            output_fields=["id", "vector"],
        ):
            vectors[str(record["id"])] = np.asarray(record["vector"], dtype=np.float32)
    missing = set(ids) - vectors.keys()
    if missing:
        raise RuntimeError(f"Milvus did not return {len(missing)} requested vectors")
    return vectors


def _build_groups(store, collection, manifest_path, threshold):
    manifest = read_cluster_manifest(manifest_path)
    records = [record for rows in manifest.values() for record in rows]
    vectors = _fetch_vectors(store, collection, [row["source_record_id"] for row in records])
    groups = []
    for cluster_id in sorted(manifest):
        terms = [
            UniqueTerm(
                text=row["text"],
                normalized_text=row["normalized_text"],
                source_record_id=row["source_record_id"],
                mention_count=int(row["mention_count"]),
                cluster_id=cluster_id,
                vector=vectors[row["source_record_id"]],
            )
            for row in manifest[cluster_id]
        ]
        groups.extend(consolidate_cluster(terms, threshold))
    return groups


def _load_entity_mentions(connection, entity_type):
    by_text = defaultdict(list)
    with connection.cursor() as cursor:
        cursor.execute(
            """SELECT entity_mention_id, entity_name FROM entity_mentions
               WHERE entity_type = %s ORDER BY entity_mention_id""",
            (entity_type,),
        )
        for mention_id, name in cursor:
            by_text[normalize_text(name)].append((mention_id, name))
    return by_text


def _load_predicate_mentions(connection):
    by_text = defaultdict(list)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT relation_mention_id, predicate FROM relation_mentions ORDER BY relation_mention_id"
        )
        for mention_id, name in cursor:
            by_text[normalize_text(name)].append((mention_id, name))
    return by_text


def _summarize(name, groups):
    unique_terms = sum(len(group.members) for group in groups)
    merged_terms = sum(len(group.members) - 1 for group in groups)
    exact_mentions = sum(
        max(0, sum(term.mention_count for term in group.members) - len(group.members))
        for group in groups
    )
    return {
        "group": name,
        "canonical_records": len(groups),
        "unique_terms": unique_terms,
        "cosine_merged_unique_terms": merged_terms,
        "additional_exact_duplicate_mentions": exact_mentions,
    }


def _write_entities(connection, run_id, entity_type, groups, mentions_by_text):
    expected_terms = {
        term.normalized_text for group in groups for term in group.members
    }
    missing_terms = expected_terms - mentions_by_text.keys()
    if missing_terms:
        examples = sorted(missing_terms)[:5]
        raise RuntimeError(
            f"PostgreSQL is missing {len(missing_terms)} normalized {entity_type} terms; "
            f"examples={examples!r}"
        )
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE entity_mentions SET canonical_entity_id=NULL, canonical_match_kind=NULL, canonical_similarity=NULL WHERE entity_type=%s",
            (entity_type,),
        )
        cursor.execute("DELETE FROM canonical_entities WHERE entity_type=%s", (entity_type,))
        for group in groups:
            rep = group.representative
            rep_mentions = mentions_by_text.get(rep.normalized_text, [])
            if not rep_mentions:
                raise RuntimeError(f"No PostgreSQL mention for representative {rep.text!r}")
            representative_mention_id = rep_mentions[0][0]
            canonical = canonical_id("entity", entity_type, rep.normalized_text)
            member_count = sum(len(mentions_by_text.get(term.normalized_text, [])) for term in group.members)
            cursor.execute(
                """INSERT INTO canonical_entities(
                       canonical_entity_id, entity_type, canonical_name, normalized_name,
                       representative_entity_mention_id, consolidation_run_id, member_count, metadata
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)""",
                (canonical, entity_type, rep.text, rep.normalized_text,
                 representative_mention_id, run_id, member_count,
                 json.dumps({"cluster_id": rep.cluster_id, "algorithm": ALGORITHM_VERSION})),
            )
            for term, similarity in zip(group.members, group.similarities, strict=True):
                for mention_id, _ in mentions_by_text.get(term.normalized_text, []):
                    if mention_id == representative_mention_id:
                        kind = "representative"
                    elif term.normalized_text == rep.normalized_text:
                        kind = "exact"
                    else:
                        kind = "cosine"
                    cursor.execute(
                        """UPDATE entity_mentions SET canonical_entity_id=%s,
                               canonical_match_kind=%s, canonical_similarity=%s
                           WHERE entity_mention_id=%s""",
                        (canonical, kind, 1.0 if kind in ("representative", "exact") else similarity, mention_id),
                    )


def _write_predicates(connection, run_id, groups, mentions_by_text):
    expected_terms = {
        term.normalized_text for group in groups for term in group.members
    }
    missing_terms = expected_terms - mentions_by_text.keys()
    if missing_terms:
        examples = sorted(missing_terms)[:5]
        raise RuntimeError(
            f"PostgreSQL is missing {len(missing_terms)} normalized predicate terms; "
            f"examples={examples!r}"
        )
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE relation_mentions SET canonical_predicate_id=NULL, canonical_match_kind=NULL, canonical_similarity=NULL"
        )
        cursor.execute("DELETE FROM canonical_predicates")
        for group in groups:
            rep = group.representative
            rep_mentions = mentions_by_text.get(rep.normalized_text, [])
            if not rep_mentions:
                raise RuntimeError(f"No PostgreSQL relation for predicate {rep.text!r}")
            representative_mention_id = rep_mentions[0][0]
            canonical = canonical_id("predicate", "predicate", rep.normalized_text)
            member_count = sum(len(mentions_by_text.get(term.normalized_text, [])) for term in group.members)
            cursor.execute(
                """INSERT INTO canonical_predicates(
                       canonical_predicate_id, canonical_name, normalized_name,
                       representative_relation_mention_id, consolidation_run_id, member_count, metadata
                   ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)""",
                (canonical, rep.text, rep.normalized_text, representative_mention_id,
                 run_id, member_count,
                 json.dumps({"cluster_id": rep.cluster_id, "algorithm": ALGORITHM_VERSION})),
            )
            for term, similarity in zip(group.members, group.similarities, strict=True):
                for mention_id, _ in mentions_by_text.get(term.normalized_text, []):
                    if mention_id == representative_mention_id:
                        kind = "representative"
                    elif term.normalized_text == rep.normalized_text:
                        kind = "exact"
                    else:
                        kind = "cosine"
                    cursor.execute(
                        """UPDATE relation_mentions SET canonical_predicate_id=%s,
                               canonical_match_kind=%s, canonical_similarity=%s
                           WHERE relation_mention_id=%s""",
                        (canonical, kind, 1.0 if kind in ("representative", "exact") else similarity, mention_id),
                    )


def main():
    args = _parser().parse_args()
    if not -1 < args.threshold < 1:
        raise SystemExit("--threshold must be between -1 and 1")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    paths = [args.cluster_dir / f"{t}_clusters.jsonl" for t in SEED_ELIGIBLE_ENTITY_TYPES]
    paths.append(args.cluster_dir / "predicate_clusters.jsonl")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing cluster manifests: " + ", ".join(missing))

    store = KnowledgeGraphVectorStore(MilvusConfig())
    entity_groups = {}
    summaries = []
    for entity_type in SEED_ELIGIBLE_ENTITY_TYPES:
        logger.info("Consolidating %s candidates", entity_type)
        groups = _build_groups(
            store, store.config.entity_collection,
            args.cluster_dir / f"{entity_type}_clusters.jsonl", args.threshold,
        )
        entity_groups[entity_type] = groups
        summaries.append(_summarize(entity_type, groups))
    logger.info("Consolidating predicate candidates")
    predicate_groups = _build_groups(
        store, store.config.predicate_collection,
        args.cluster_dir / "predicate_clusters.jsonl", args.threshold,
    )
    summaries.append(_summarize("predicate", predicate_groups))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Dry run only; rerun with --apply to write PostgreSQL mappings.")
        return 0

    run_id = uuid.uuid4()
    digest = manifest_sha256(paths)
    with psycopg.connect(DatabaseSettings().database_url) as connection:
        create_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO entity_consolidation_runs(
                       run_id,status,algorithm_version,similarity_threshold,cluster_manifest_sha256
                   ) VALUES (%s,'running',%s,%s,%s)""",
                (run_id, ALGORITHM_VERSION, args.threshold, digest),
            )
        for entity_type, groups in entity_groups.items():
            _write_entities(
                connection, run_id, entity_type, groups,
                _load_entity_mentions(connection, entity_type),
            )
        _write_predicates(connection, run_id, predicate_groups, _load_predicate_mentions(connection))
        with connection.cursor() as cursor:
            cursor.execute(
                """UPDATE entity_consolidation_runs
                   SET status='completed', completed_at=now(), summary=%s::jsonb WHERE run_id=%s""",
                (json.dumps(summaries), run_id),
            )
        connection.commit()
    print(f"PostgreSQL consolidation committed successfully; run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
