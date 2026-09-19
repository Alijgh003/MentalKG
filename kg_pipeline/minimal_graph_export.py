"""Export the normalized DSM data into a minimal four-table graph database."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo


UNRESOLVED_ENTITY_NAMESPACE = uuid.UUID("a14f9d0d-327c-46af-a527-cdfd616c8103")


SCHEMA_SQL = """
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL
);

CREATE TABLE entities (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    type TEXT
);

CREATE TABLE facts (
    id UUID PRIMARY KEY,
    subject_id UUID NOT NULL REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_id UUID NOT NULL REFERENCES entities(id)
);

CREATE TABLE mentions (
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, chunk_id)
);

CREATE INDEX idx_facts_subject ON facts(subject_id);
CREATE INDEX idx_facts_object ON facts(object_id);
CREATE INDEX idx_mentions_chunk ON mentions(chunk_id);
"""


def chunk_content(content: str | None, hierarchy: str | None) -> str:
    """Store exactly source content followed by its hierarchy context."""
    parts = [part.strip() for part in (content, hierarchy) if part and part.strip()]
    return "\n\n".join(parts)


def unresolved_entity_id(relation_id: uuid.UUID, role: str) -> uuid.UUID:
    return uuid.uuid5(UNRESOLVED_ENTITY_NAMESPACE, f"{relation_id}:{role}")


def target_dsn(source_dsn: str, database_name: str) -> str:
    params = conninfo_to_dict(source_dsn)
    params["dbname"] = database_name
    return make_conninfo(**params)


def ensure_database(source_dsn: str, database_name: str) -> str:
    params = conninfo_to_dict(source_dsn)
    source_database = params.get("dbname") or params.get("database")
    if database_name == source_database:
        raise ValueError("Target database must differ from the source database")
    admin_params = dict(params)
    admin_params["dbname"] = "postgres"
    with psycopg.connect(make_conninfo(**admin_params), autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (database_name,))
            if cursor.fetchone() is None:
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
    return target_dsn(source_dsn, database_name)


@dataclass(frozen=True)
class ExportSummary:
    chunks: int
    canonical_entities: int
    standalone_entities: int
    unresolved_entities: int
    entities: int
    facts: int
    mentions: int


class MinimalGraphExporter:
    def __init__(self, source_connection, target_connection):
        self.source = source_connection
        self.target = target_connection

    def rebuild(self) -> ExportSummary:
        self._reset_target_schema()
        chunk_count = self._copy_chunks()
        mention_to_entity, canonical_count, standalone_count = self._copy_known_entities()
        facts, unresolved = self._prepare_facts(mention_to_entity)
        self._copy_unresolved_entities(unresolved)
        self._copy_facts_and_mentions(facts)
        summary = ExportSummary(
            chunks=chunk_count,
            canonical_entities=canonical_count,
            standalone_entities=standalone_count,
            unresolved_entities=len(unresolved),
            entities=canonical_count + standalone_count + len(unresolved),
            facts=len(facts),
            mentions=len(facts),
        )
        self._validate(summary)
        return summary

    def _reset_target_schema(self):
        with self.target.cursor() as cursor:
            cursor.execute("DROP SCHEMA public CASCADE")
            cursor.execute("CREATE SCHEMA public")
            cursor.execute(SCHEMA_SQL)

    def _copy_chunks(self) -> int:
        with self.source.cursor() as source_cursor:
            source_cursor.execute(
                """SELECT node_id,content,hierarchy_path
                   FROM nodes WHERE is_selected AND tree_role='leaf' ORDER BY node_id"""
            )
            rows = source_cursor.fetchall()
        with self.target.cursor() as cursor, cursor.copy(
            "COPY chunks(id,content) FROM STDIN"
        ) as copy:
            for node_id, content, hierarchy in rows:
                copy.write_row((node_id, chunk_content(content, hierarchy)))
        return len(rows)

    def _copy_known_entities(self):
        mention_to_entity = {}
        with self.source.cursor() as cursor:
            cursor.execute(
                """SELECT canonical_entity_id,canonical_name,entity_type
                   FROM canonical_entities ORDER BY canonical_entity_id"""
            )
            canonical_rows = cursor.fetchall()
            cursor.execute(
                """SELECT entity_mention_id,entity_name,entity_type
                   FROM entity_mentions WHERE canonical_entity_id IS NULL
                   ORDER BY entity_mention_id"""
            )
            standalone_rows = cursor.fetchall()
            cursor.execute(
                "SELECT entity_mention_id,canonical_entity_id FROM entity_mentions"
            )
            for mention_id, canonical_id in cursor:
                mention_to_entity[mention_id] = canonical_id or mention_id

        with self.target.cursor() as cursor, cursor.copy(
            "COPY entities(id,text,type) FROM STDIN"
        ) as copy:
            for entity_id, text, entity_type in canonical_rows:
                copy.write_row((entity_id, text, entity_type))
            for entity_id, text, entity_type in standalone_rows:
                copy.write_row((entity_id, text, entity_type))
        return mention_to_entity, len(canonical_rows), len(standalone_rows)

    def _prepare_facts(self, mention_to_entity):
        with self.source.cursor() as cursor:
            cursor.execute(
                """SELECT r.relation_mention_id,r.node_id,r.subject_text,r.predicate,r.object_text,
                          sl.entity_mention_id,ol.entity_mention_id
                   FROM relation_mentions r
                   LEFT JOIN relation_entity_links sl ON sl.relation_mention_id=r.relation_mention_id
                        AND sl.endpoint_role='subject'
                   LEFT JOIN relation_entity_links ol ON ol.relation_mention_id=r.relation_mention_id
                        AND ol.endpoint_role='object'
                   ORDER BY r.relation_mention_id"""
            )
            rows = cursor.fetchall()
        facts = []
        unresolved = {}
        for relation_id, node_id, subject_text, predicate, object_text, subject_mention, object_mention in rows:
            if subject_mention is None:
                subject_id = unresolved_entity_id(relation_id, "subject")
                unresolved[subject_id] = subject_text
            else:
                subject_id = mention_to_entity[subject_mention]
            if object_mention is None:
                object_id = unresolved_entity_id(relation_id, "object")
                unresolved[object_id] = object_text
            else:
                object_id = mention_to_entity[object_mention]
            facts.append((relation_id, subject_id, predicate, object_id, node_id))
        return facts, unresolved

    def _copy_unresolved_entities(self, unresolved):
        with self.target.cursor() as cursor, cursor.copy(
            "COPY entities(id,text,type) FROM STDIN"
        ) as copy:
            for entity_id, text in sorted(unresolved.items(), key=lambda item: str(item[0])):
                copy.write_row((entity_id, text, "unresolved"))

    def _copy_facts_and_mentions(self, facts):
        with self.target.cursor() as cursor, cursor.copy(
            "COPY facts(id,subject_id,predicate,object_id) FROM STDIN"
        ) as copy:
            for relation_id, subject_id, predicate, object_id, _ in facts:
                copy.write_row((relation_id, subject_id, predicate, object_id))
        with self.target.cursor() as cursor, cursor.copy(
            "COPY mentions(fact_id,chunk_id) FROM STDIN"
        ) as copy:
            for relation_id, _, _, _, node_id in facts:
                copy.write_row((relation_id, node_id))

    def _validate(self, summary: ExportSummary):
        with self.target.cursor() as cursor:
            cursor.execute(
                """SELECT
                     (SELECT count(*) FROM chunks),
                     (SELECT count(*) FROM entities),
                     (SELECT count(*) FROM facts),
                     (SELECT count(*) FROM mentions),
                     (SELECT count(*) FROM entities WHERE type='unresolved'),
                     (SELECT count(*) FROM facts f LEFT JOIN entities e ON e.id=f.subject_id WHERE e.id IS NULL),
                     (SELECT count(*) FROM facts f LEFT JOIN entities e ON e.id=f.object_id WHERE e.id IS NULL),
                     (SELECT count(*) FROM mentions m LEFT JOIN chunks c ON c.id=m.chunk_id WHERE c.id IS NULL)"""
            )
            counts = cursor.fetchone()
            cursor.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
            )
            tables = [row[0] for row in cursor.fetchall()]
        expected_counts = (
            summary.chunks, summary.entities, summary.facts, summary.mentions,
            summary.unresolved_entities, 0, 0, 0,
        )
        if counts != expected_counts:
            raise RuntimeError(f"Target validation failed: counts={counts}, expected={expected_counts}")
        if tables != ["chunks", "entities", "facts", "mentions"]:
            raise RuntimeError(f"Target must contain exactly four tables, found {tables}")


def database_name_from_env() -> str:
    return os.getenv("DSM_MINIMAL_GRAPH_DATABASE_NAME", "dsm5_minimal_graph")
