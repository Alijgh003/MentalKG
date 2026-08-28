"""PostgreSQL DDL for source documents, tree passages, and KG extractions."""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_sources (
    source_path TEXT PRIMARY KEY,
    source_role TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_size BIGINT NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    document_code TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pages (
    document_code TEXT NOT NULL REFERENCES documents(document_code),
    page_number INTEGER NOT NULL CHECK (page_number >= 0),
    raw_markdown TEXT,
    raw_payload JSONB NOT NULL,
    PRIMARY KEY (document_code, page_number)
);

CREATE TABLE IF NOT EXISTS page_parse_results (
    document_code TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    parsed_content TEXT,
    page_tree JSONB,
    llm_metadata JSONB,
    raw_payload JSONB NOT NULL,
    source_path TEXT NOT NULL REFERENCES ingestion_sources(source_path),
    PRIMARY KEY (document_code, page_number),
    FOREIGN KEY (document_code, page_number)
        REFERENCES pages(document_code, page_number)
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    parent_node_id TEXT REFERENCES nodes(node_id) DEFERRABLE INITIALLY DEFERRED,
    document_code TEXT NOT NULL REFERENCES documents(document_code),
    page_number INTEGER NOT NULL CHECK (page_number >= 0),
    node_kind TEXT,
    tree_role TEXT NOT NULL CHECK (tree_role IN ('leaf', 'parent')),
    heading TEXT,
    content TEXT,
    section_name TEXT,
    hierarchy_path TEXT,
    is_selected BOOLEAN NOT NULL DEFAULT FALSE,
    is_derived BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (document_code, page_number)
        REFERENCES pages(document_code, page_number)
);

CREATE TABLE IF NOT EXISTS node_source_records (
    source_path TEXT NOT NULL REFERENCES ingestion_sources(source_path),
    node_id TEXT NOT NULL REFERENCES nodes(node_id) DEFERRABLE INITIALLY DEFERRED,
    raw_payload JSONB NOT NULL,
    PRIMARY KEY (source_path, node_id)
);

CREATE TABLE IF NOT EXISTS node_page_spans (
    node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    document_code TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    span_role TEXT NOT NULL CHECK (span_role IN ('primary', 'continuation')),
    PRIMARY KEY (node_id, document_code, page_number, span_role),
    FOREIGN KEY (document_code, page_number)
        REFERENCES pages(document_code, page_number)
);

CREATE TABLE IF NOT EXISTS derived_node_sources (
    derived_node_id TEXT PRIMARY KEY
        REFERENCES nodes(node_id) ON DELETE CASCADE,
    original_node_id TEXT NOT NULL REFERENCES nodes(node_id),
    continuation_document_code TEXT NOT NULL,
    continuation_page_number INTEGER NOT NULL,
    continuation_content TEXT,
    FOREIGN KEY (continuation_document_code, continuation_page_number)
        REFERENCES pages(document_code, page_number)
);

CREATE TABLE IF NOT EXISTS node_ingestion_issues (
    source_path TEXT NOT NULL REFERENCES ingestion_sources(source_path),
    node_id TEXT NOT NULL REFERENCES nodes(node_id),
    issue_kind TEXT NOT NULL,
    raw_payload JSONB NOT NULL,
    PRIMARY KEY (source_path, node_id, issue_kind)
);

CREATE TABLE IF NOT EXISTS kg_source_records (
    source_path TEXT NOT NULL REFERENCES ingestion_sources(source_path),
    record_ordinal INTEGER NOT NULL,
    node_id TEXT NOT NULL REFERENCES nodes(node_id),
    extraction_set TEXT NOT NULL,
    artifact_kind TEXT NOT NULL CHECK (artifact_kind IN ('combined', 'entities', 'relations')),
    raw_payload JSONB NOT NULL,
    PRIMARY KEY (source_path, record_ordinal)
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_mention_id UUID PRIMARY KEY,
    extraction_set TEXT NOT NULL,
    node_id TEXT NOT NULL REFERENCES nodes(node_id),
    source_path TEXT NOT NULL REFERENCES ingestion_sources(source_path),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    entity_name TEXT NOT NULL,
    entity_type TEXT,
    start_char INTEGER,
    end_char INTEGER,
    raw_payload JSONB NOT NULL,
    UNIQUE (extraction_set, node_id, source_path, ordinal)
);

CREATE TABLE IF NOT EXISTS relation_mentions (
    relation_mention_id UUID PRIMARY KEY,
    extraction_set TEXT NOT NULL,
    node_id TEXT NOT NULL REFERENCES nodes(node_id),
    source_path TEXT NOT NULL REFERENCES ingestion_sources(source_path),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    subject_text TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_text TEXT NOT NULL,
    metadata JSONB,
    raw_payload JSONB NOT NULL,
    UNIQUE (extraction_set, node_id, source_path, ordinal)
);

CREATE TABLE IF NOT EXISTS relation_entity_links (
    relation_mention_id UUID NOT NULL
        REFERENCES relation_mentions(relation_mention_id) ON DELETE CASCADE,
    entity_mention_id UUID NOT NULL
        REFERENCES entity_mentions(entity_mention_id) ON DELETE CASCADE,
    endpoint_role TEXT NOT NULL CHECK (endpoint_role IN ('subject', 'object')),
    PRIMARY KEY (relation_mention_id, endpoint_role)
);

CREATE INDEX IF NOT EXISTS idx_nodes_document_page
    ON nodes(document_code, page_number);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_node ON entity_mentions(node_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_name_lower ON entity_mentions(lower(entity_name));
CREATE INDEX IF NOT EXISTS idx_relation_mentions_node ON relation_mentions(node_id);
CREATE INDEX IF NOT EXISTS idx_relation_mentions_predicate ON relation_mentions(predicate);
"""


def create_schema(connection) -> None:
    """Create all tables and indexes in an already-created PostgreSQL database."""
    logger.info("Creating or updating PostgreSQL schema")
    with connection.cursor() as cursor:
        cursor.execute(SCHEMA_SQL)
    logger.info("PostgreSQL schema is ready")
