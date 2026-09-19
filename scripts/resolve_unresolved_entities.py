#!/usr/bin/env python3
"""Resolve minimal-graph entities against the canonical Milvus entity index."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DatabaseSettings
from kg_pipeline.embeddings import EmbeddingConfig, OpenAICompatibleEmbedder
from kg_pipeline.minimal_graph_export import database_name_from_env, target_dsn
from kg_pipeline.vector_store import KnowledgeGraphVectorStore, MilvusConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Resolution:
    unresolved_id: str
    unresolved_text: str
    resolved_id: str
    resolved_text: str
    resolved_type: str
    similarity: float


def batches(items: list[tuple[str, str]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed unresolved entities, find their nearest canonical Milvus entity, "
            "and optionally rewrite facts and delete resolved placeholders."
        )
    )
    parser.add_argument("--execute", action="store_true", help="Apply changes; default is dry-run")
    parser.add_argument("--threshold", type=float, default=0.87)
    parser.add_argument("--limit", type=int, help="Process only this many unresolved entities")
    parser.add_argument("--search-batch-size", type=int, default=256)
    parser.add_argument("--show", type=int, default=20, help="Print this many accepted matches")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not -1.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between -1 and 1")
    if args.search_batch_size <= 0:
        parser.error("--search-batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.show < 0:
        parser.error("--show cannot be negative")
    return args


def load_unresolved(connection, limit: int | None) -> list[tuple[str, str]]:
    query = "SELECT id::text,text FROM entities WHERE type='unresolved' ORDER BY id"
    params: tuple[int, ...] = ()
    if limit is not None:
        query += " LIMIT %s"
        params = (limit,)
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        return [(row[0], row[1]) for row in cursor]


def find_resolutions(
    unresolved: list[tuple[str, str]],
    *,
    embedder: OpenAICompatibleEmbedder,
    store: KnowledgeGraphVectorStore,
    threshold: float,
    search_batch_size: int,
) -> tuple[list[Resolution], int]:
    accepted: list[Resolution] = []
    rejected = 0
    collection = store.config.canonical_entity_collection
    dimension = store.config.canonical_entity_dimension

    for batch_number, batch in enumerate(batches(unresolved, search_batch_size), start=1):
        texts = [text for _, text in batch]
        vectors = embedder.embed(texts, dimension=dimension)
        results = store.search(
            collection,
            vectors.tolist(),
            limit=1,
            output_fields=("text", "entity_type", "embedding_model"),
        )
        if len(results) != len(batch):
            raise RuntimeError(
                f"Milvus returned {len(results)} result lists for a batch of {len(batch)}"
            )
        for (unresolved_id, unresolved_text), hits in zip(batch, results, strict=True):
            if not hits:
                rejected += 1
                continue
            hit = hits[0]
            entity = hit.get("entity", {})
            hit_model = str(entity.get("embedding_model") or "")
            if hit_model and hit_model != embedder.config.model:
                raise RuntimeError(
                    "Embedding model mismatch: query model "
                    f"{embedder.config.model!r}, collection hit model {hit_model!r}"
                )
            similarity = float(hit.get("distance", 0.0))
            if similarity <= threshold:  # Requirement is strictly greater than the threshold.
                rejected += 1
                continue
            accepted.append(
                Resolution(
                    unresolved_id=unresolved_id,
                    unresolved_text=unresolved_text,
                    resolved_id=str(hit["id"]),
                    resolved_text=str(entity.get("text") or ""),
                    resolved_type=str(entity.get("entity_type") or ""),
                    similarity=similarity,
                )
            )
        logger.info(
            "Searched batch %s: processed=%s accepted=%s",
            batch_number,
            min(batch_number * search_batch_size, len(unresolved)),
            len(accepted),
        )
    return accepted, rejected


def apply_resolutions(connection, resolutions: list[Resolution]) -> dict[str, int]:
    if not resolutions:
        return {"subject_references_updated": 0, "object_references_updated": 0, "entities_deleted": 0}

    target_ids = {resolution.resolved_id for resolution in resolutions}
    with connection.cursor() as cursor:
        cursor.execute("SELECT id::text FROM entities WHERE id = ANY(%s::uuid[])", (list(target_ids),))
        existing_target_ids = {row[0] for row in cursor}
        missing = sorted(target_ids - existing_target_ids)
        if missing:
            raise RuntimeError(
                f"Refusing to update: {len(missing)} Milvus target IDs do not exist in PostgreSQL"
            )

        cursor.execute(
            "CREATE TEMP TABLE entity_resolutions "
            "(unresolved_id UUID PRIMARY KEY, resolved_id UUID NOT NULL) ON COMMIT DROP"
        )
        with cursor.copy(
            "COPY entity_resolutions(unresolved_id,resolved_id) FROM STDIN"
        ) as copy:
            for resolution in resolutions:
                copy.write_row((resolution.unresolved_id, resolution.resolved_id))

        cursor.execute(
            "UPDATE facts f SET subject_id=r.resolved_id "
            "FROM entity_resolutions r WHERE f.subject_id=r.unresolved_id"
        )
        subjects_updated = cursor.rowcount
        cursor.execute(
            "UPDATE facts f SET object_id=r.resolved_id "
            "FROM entity_resolutions r WHERE f.object_id=r.unresolved_id"
        )
        objects_updated = cursor.rowcount
        cursor.execute(
            "DELETE FROM entities e USING entity_resolutions r "
            "WHERE e.id=r.unresolved_id AND e.type='unresolved' "
            "AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.subject_id=e.id OR f.object_id=e.id)"
        )
        deleted = cursor.rowcount

    if deleted != len(resolutions):
        raise RuntimeError(
            f"Expected to delete {len(resolutions)} resolved placeholders, deleted {deleted}"
        )
    return {
        "subject_references_updated": subjects_updated,
        "object_references_updated": objects_updated,
        "entities_deleted": deleted,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    source_dsn = DatabaseSettings().database_url
    destination_dsn = target_dsn(source_dsn, database_name_from_env())
    embedding_config = EmbeddingConfig.from_env()
    milvus_config = MilvusConfig.from_env()
    embedder = OpenAICompatibleEmbedder(embedding_config)
    store = KnowledgeGraphVectorStore(milvus_config)

    collection = milvus_config.canonical_entity_collection
    if not store.client.has_collection(collection_name=collection):
        raise RuntimeError(f"Milvus collection does not exist: {collection}")
    store.client.load_collection(collection_name=collection)

    with psycopg.connect(destination_dsn) as connection:
        unresolved = load_unresolved(connection, args.limit)
        resolutions, rejected = find_resolutions(
            unresolved,
            embedder=embedder,
            store=store,
            threshold=args.threshold,
            search_batch_size=args.search_batch_size,
        )

        print(json.dumps({
            "database": database_name_from_env(),
            "milvus_collection": collection,
            "embedding_model": embedding_config.model,
            "threshold_rule": f"cosine_similarity > {args.threshold}",
            "unresolved_processed": len(unresolved),
            "accepted": len(resolutions),
            "rejected_or_no_hit": rejected,
            "execute": args.execute,
        }, ensure_ascii=False, indent=2))
        for resolution in resolutions[: args.show]:
            print(
                f"{resolution.similarity:.6f} | {resolution.unresolved_text!r} -> "
                f"{resolution.resolved_text!r} [{resolution.resolved_type}] "
                f"({resolution.unresolved_id} -> {resolution.resolved_id})"
            )

        if not args.execute:
            connection.rollback()
            return 0

        changes = apply_resolutions(connection, resolutions)
        connection.commit()

    with psycopg.connect(destination_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM entities WHERE type='unresolved'")
            unresolved_remaining = cursor.fetchone()[0]
            cursor.execute(
                "SELECT count(*) FROM facts f "
                "LEFT JOIN entities s ON s.id=f.subject_id "
                "LEFT JOIN entities o ON o.id=f.object_id "
                "WHERE s.id IS NULL OR o.id IS NULL"
            )
            broken_fact_references = cursor.fetchone()[0]

    print(json.dumps({
        **changes,
        "unresolved_remaining": unresolved_remaining,
        "broken_fact_references": broken_fact_references,
    }, ensure_ascii=False, indent=2))
    return 0 if broken_fact_references == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
