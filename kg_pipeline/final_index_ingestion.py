"""Build final Milvus retrieval indexes from the consolidated PostgreSQL KG."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from .clustering import SEED_ELIGIBLE_ENTITY_TYPES
from .embeddings import OpenAICompatibleEmbedder
from .vector_store import KnowledgeGraphVectorStore

logger = logging.getLogger(__name__)


MAX_MILVUS_TEXT_LENGTH = 65_535


def _batches(items: Iterable, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _clip(value: str, limit: int = MAX_MILVUS_TEXT_LENGTH) -> str:
    return value if len(value) <= limit else value[:limit]


def _chunk_text(*, hierarchy_path: str, heading: str, section_name: str, content: str) -> str:
    parts = [
        hierarchy_path,
        heading if not hierarchy_path else "",
        section_name if not hierarchy_path and not heading else "",
        content,
    ]
    return _clip("\n".join(part for part in parts if part).strip())


class FinalIndexIngestion:
    """Resumable PostgreSQL-to-Milvus ingestion for final retrieval indexes."""

    def __init__(
        self,
        pg_connection,
        embedder: OpenAICompatibleEmbedder,
        store: KnowledgeGraphVectorStore,
    ):
        self.pg_connection = pg_connection
        self.embedder = embedder
        self.store = store
        self._vector_cache: dict[str, list[float]] = {}

    def ingest_canonical_entities(self, limit: int | None = None) -> int:
        query = """SELECT canonical_entity_id::text, canonical_name, normalized_name,
                          entity_type, member_count
                   FROM canonical_entities
                   WHERE entity_type = ANY(%s)
                   ORDER BY canonical_entity_id"""
        params: tuple[Any, ...] = (list(SEED_ELIGIBLE_ENTITY_TYPES),)
        if limit is not None:
            query += " LIMIT %s"
            params = (*params, limit)
        collection = self.store.config.canonical_entity_collection
        dimension = self.store.config.canonical_entity_dimension
        inserted = 0
        with self.pg_connection.cursor(name="canonical_entity_embedding_source") as cursor:
            cursor.execute(query, params)
            for batch in _batches(cursor, self.store.config.insert_batch_size):
                missing = self.store.missing_ids(collection, [row[0] for row in batch])
                rows = [row for row in batch if row[0] in missing]
                if not rows:
                    continue
                texts = [_clean(row[1]) for row in rows]
                self._ensure_vectors(texts, dimension)
                payload = [
                    {
                        "id": row[0],
                        "vector": self._vector_cache[f"{dimension}:{_clean(row[1])}"],
                        "text": _clean(row[1]),
                        "normalized_text": _clean(row[2]),
                        "entity_type": _clean(row[3]),
                        "canonical_name": _clean(row[1]),
                        "member_count": int(row[4]),
                        "seed_eligible": True,
                        "embedding_model": self.embedder.config.model,
                    }
                    for row in rows
                ]
                self.store.upsert(collection, payload)
                inserted += len(payload)
                logger.info("Canonical entity vectors inserted: %s", inserted)
        return inserted

    def ingest_canonical_predicates(self, limit: int | None = None) -> int:
        query = """SELECT canonical_predicate_id::text, canonical_name,
                          normalized_name, member_count
                   FROM canonical_predicates ORDER BY canonical_predicate_id"""
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT %s"
            params = (limit,)
        collection = self.store.config.canonical_predicate_collection
        dimension = self.store.config.canonical_predicate_dimension
        inserted = 0
        with self.pg_connection.cursor(name="canonical_predicate_embedding_source") as cursor:
            cursor.execute(query, params)
            for batch in _batches(cursor, self.store.config.insert_batch_size):
                missing = self.store.missing_ids(collection, [row[0] for row in batch])
                rows = [row for row in batch if row[0] in missing]
                if not rows:
                    continue
                texts = [_clean(row[1]) for row in rows]
                self._ensure_vectors(texts, dimension)
                payload = [
                    {
                        "id": row[0],
                        "vector": self._vector_cache[f"{dimension}:{_clean(row[1])}"],
                        "text": _clean(row[1]),
                        "normalized_text": _clean(row[2]),
                        "canonical_name": _clean(row[1]),
                        "member_count": int(row[3]),
                        "embedding_model": self.embedder.config.model,
                    }
                    for row in rows
                ]
                self.store.upsert(collection, payload)
                inserted += len(payload)
                logger.info("Canonical predicate vectors inserted: %s", inserted)
        return inserted

    def ingest_raw_triples(self, limit: int | None = None) -> int:
        query = """SELECT relation_mention_id::text, subject_text, predicate,
                          object_text, node_id, extraction_set, source_path, ordinal
                   FROM relation_mentions ORDER BY relation_mention_id"""
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT %s"
            params = (limit,)
        collection = self.store.config.triple_collection
        dimension = self.store.config.triple_dimension
        inserted = 0
        with self.pg_connection.cursor(name="raw_triple_embedding_source") as cursor:
            cursor.execute(query, params)
            for batch in _batches(cursor, self.store.config.insert_batch_size):
                missing = self.store.missing_ids(collection, [row[0] for row in batch])
                rows = [row for row in batch if row[0] in missing]
                if not rows:
                    continue
                texts = [
                    f"{_clean(row[1])} {_clean(row[2])} {_clean(row[3])}"
                    for row in rows
                ]
                self._ensure_vectors(texts, dimension)
                payload = [
                    {
                        "id": row[0],
                        "vector": self._vector_cache[f"{dimension}:{text}"],
                        "text": text,
                        "subject_text": _clip(_clean(row[1]), 8192),
                        "predicate": _clip(_clean(row[2]), 2048),
                        "object_text": _clip(_clean(row[3]), 8192),
                        "node_id": _clean(row[4]),
                        "extraction_set": _clean(row[5]),
                        "source_path": _clean(row[6]),
                        "ordinal": int(row[7]),
                        "embedding_model": self.embedder.config.model,
                    }
                    for row, text in zip(rows, texts, strict=True)
                ]
                self.store.upsert(collection, payload)
                inserted += len(payload)
                logger.info("Raw triple vectors inserted: %s", inserted)
        return inserted

    def ingest_source_chunks(self, limit: int | None = None) -> int:
        query = """SELECT node_id, document_code, page_number, heading, content,
                          section_name, hierarchy_path, is_derived
                   FROM nodes
                   WHERE is_selected AND tree_role = 'leaf'
                   ORDER BY node_id"""
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT %s"
            params = (limit,)
        collection = self.store.config.source_chunk_collection
        dimension = self.store.config.source_chunk_dimension
        inserted = 0
        with self.pg_connection.cursor(name="source_chunk_embedding_source") as cursor:
            cursor.execute(query, params)
            for batch in _batches(cursor, self.store.config.insert_batch_size):
                missing = self.store.missing_ids(collection, [row[0] for row in batch])
                rows = [row for row in batch if row[0] in missing]
                if not rows:
                    continue
                texts = [
                    _chunk_text(
                        hierarchy_path=_clean(row[6]),
                        heading=_clean(row[3]),
                        section_name=_clean(row[5]),
                        content=_clean(row[4]),
                    ) or f"Node: {row[0]}"
                    for row in rows
                ]
                self._ensure_vectors(texts, dimension)
                payload = [
                    {
                        "id": row[0],
                        "vector": self._vector_cache[f"{dimension}:{text}"],
                        "text": text,
                        "node_id": row[0],
                        "document_code": _clean(row[1]),
                        "page_number": int(row[2]),
                        "heading": _clip(_clean(row[3]), 4096),
                        "section_name": _clip(_clean(row[5]), 4096),
                        "hierarchy_path": _clip(_clean(row[6]), 16384),
                        "is_derived": bool(row[7]),
                        "embedding_model": self.embedder.config.model,
                    }
                    for row, text in zip(rows, texts, strict=True)
                ]
                self.store.upsert(collection, payload)
                inserted += len(payload)
                logger.info("Source chunk vectors inserted: %s", inserted)
        return inserted

    def _ensure_vectors(self, texts: list[str], dimension: int) -> None:
        unseen = list(
            dict.fromkeys(text for text in texts if f"{dimension}:{text}" not in self._vector_cache)
        )
        for batch in _batches(unseen, self.embedder.config.batch_size):
            vectors = self.embedder.embed(batch, dimension=dimension)
            for text, vector in zip(batch, vectors, strict=True):
                self._vector_cache[f"{dimension}:{text}"] = vector.tolist()
