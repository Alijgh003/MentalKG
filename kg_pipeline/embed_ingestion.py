"""Embed PostgreSQL entity mentions and unique predicates into Milvus."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable

from .embeddings import OpenAICompatibleEmbedder
from .vector_store import KnowledgeGraphVectorStore

logger = logging.getLogger(__name__)


def _batches(items: Iterable, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class EmbeddingIngestion:
    """Resumable PostgreSQL-to-Milvus embedding ingestion."""

    def __init__(self, pg_connection, embedder: OpenAICompatibleEmbedder, store: KnowledgeGraphVectorStore):
        self.pg_connection = pg_connection
        self.embedder = embedder
        self.store = store
        self._vector_cache: dict[str, list[float]] = {}

    def ingest_entities(self, limit: int | None = None) -> int:
        query = """SELECT entity_mention_id::text, entity_name, entity_type, node_id,
                          extraction_set, source_path, ordinal
                   FROM entity_mentions ORDER BY entity_mention_id"""
        params = ()
        if limit is not None:
            query += " LIMIT %s"
            params = (limit,)
        inserted = 0
        with self.pg_connection.cursor(name="entity_embedding_source") as cursor:
            cursor.execute(query, params)
            for batch in _batches(cursor, self.store.config.insert_batch_size):
                missing = self.store.missing_ids(
                    self.store.config.entity_collection, [row[0] for row in batch]
                )
                rows = [row for row in batch if row[0] in missing]
                if not rows:
                    continue
                self._ensure_vectors(
                    [row[1].strip() for row in rows], self.store.config.entity_dimension
                )
                payload = [
                    {
                        "id": row[0],
                        "vector": self._vector_cache[
                            f"{self.store.config.entity_dimension}:{row[1].strip()}"
                        ],
                        "text": row[1],
                        "normalized_text": row[1].strip().casefold(),
                        "entity_type": row[2] or "",
                        "node_id": row[3],
                        "extraction_set": row[4],
                        "source_path": row[5],
                        "ordinal": row[6],
                        "embedding_model": self.embedder.config.model,
                    }
                    for row in rows
                ]
                self.store.upsert(self.store.config.entity_collection, payload)
                inserted += len(payload)
                logger.info("Entity vectors inserted: %s", inserted)
        return inserted

    def ingest_predicates(self) -> int:
        with self.pg_connection.cursor() as cursor:
            cursor.execute(
                """SELECT predicate, count(*) FROM relation_mentions
                   GROUP BY predicate ORDER BY predicate"""
            )
            rows = cursor.fetchall()
        records = [
            (hashlib.sha256(predicate.strip().casefold().encode()).hexdigest(), predicate, count)
            for predicate, count in rows
        ]
        inserted = 0
        for batch in _batches(records, self.store.config.insert_batch_size):
            missing = self.store.missing_ids(
                self.store.config.predicate_collection, [row[0] for row in batch]
            )
            rows_to_insert = [row for row in batch if row[0] in missing]
            if not rows_to_insert:
                continue
            texts = [row[1].strip() for row in rows_to_insert]
            self._ensure_vectors(texts, self.store.config.predicate_dimension)
            payload = [
                {
                    "id": row[0],
                    "vector": self._vector_cache[
                        f"{self.store.config.predicate_dimension}:{row[1].strip()}"
                    ],
                    "text": row[1],
                    "normalized_text": row[1].strip().casefold(),
                    "occurrence_count": row[2],
                    "embedding_model": self.embedder.config.model,
                }
                for row in rows_to_insert
            ]
            self.store.upsert(self.store.config.predicate_collection, payload)
            inserted += len(payload)
        logger.info("Predicate vectors inserted: %s", inserted)
        return inserted

    def _ensure_vectors(self, texts: list[str], dimension: int) -> None:
        # Dimension participates in the cache key so future heterogeneous
        # collections can safely embed the same text at different sizes.
        unseen = list(
            dict.fromkeys(text for text in texts if f"{dimension}:{text}" not in self._vector_cache)
        )
        for batch in _batches(unseen, self.embedder.config.batch_size):
            vectors = self.embedder.embed(batch, dimension=dimension)
            for text, vector in zip(batch, vectors, strict=True):
                self._vector_cache[f"{dimension}:{text}"] = vector.tolist()
