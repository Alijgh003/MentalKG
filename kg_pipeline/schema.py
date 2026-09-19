"""PostgreSQL DDL for source documents, tree passages, and KG extractions."""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id UUID PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    source_root TEXT NOT NULL,
    importer_version TEXT NOT NULL,
    summary JSONB,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_sources (
    source_path TEXT PRIMARY KEY,
    source_role TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_size BIGINT NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE ingestion_sources ADD COLUMN IF NOT EXISTS last_run_id UUID
    REFERENCES ingestion_runs(run_id);

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

CREATE TABLE IF NOT EXISTS node_extraction_status (
    node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    extraction_set TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('success', 'rejected', 'empty', 'missing', 'failed')),
    has_entities BOOLEAN NOT NULL DEFAULT FALSE,
    has_relations BOOLEAN NOT NULL DEFAULT FALSE,
    detail JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (node_id, extraction_set)
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

CREATE TABLE IF NOT EXISTS entity_consolidation_runs (
    run_id UUID PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    algorithm_version TEXT NOT NULL,
    similarity_threshold DOUBLE PRECISION NOT NULL,
    cluster_manifest_sha256 TEXT NOT NULL,
    summary JSONB,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS canonical_entities (
    canonical_entity_id UUID PRIMARY KEY,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    representative_entity_mention_id UUID
        REFERENCES entity_mentions(entity_mention_id) ON DELETE SET NULL,
    consolidation_run_id UUID NOT NULL
        REFERENCES entity_consolidation_runs(run_id),
    member_count INTEGER NOT NULL CHECK (member_count > 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (entity_type, normalized_name)
);

ALTER TABLE entity_mentions ADD COLUMN IF NOT EXISTS canonical_entity_id UUID
    REFERENCES canonical_entities(canonical_entity_id) ON DELETE SET NULL;
ALTER TABLE entity_mentions ADD COLUMN IF NOT EXISTS canonical_match_kind TEXT
    CHECK (canonical_match_kind IN ('representative', 'exact', 'cosine'));
ALTER TABLE entity_mentions ADD COLUMN IF NOT EXISTS canonical_similarity DOUBLE PRECISION
    CHECK (canonical_similarity >= -1 AND canonical_similarity <= 1);

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

CREATE TABLE IF NOT EXISTS canonical_predicates (
    canonical_predicate_id UUID PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    representative_relation_mention_id UUID
        REFERENCES relation_mentions(relation_mention_id) ON DELETE SET NULL,
    consolidation_run_id UUID NOT NULL
        REFERENCES entity_consolidation_runs(run_id),
    member_count INTEGER NOT NULL CHECK (member_count > 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE relation_mentions ADD COLUMN IF NOT EXISTS canonical_predicate_id UUID
    REFERENCES canonical_predicates(canonical_predicate_id) ON DELETE SET NULL;
ALTER TABLE relation_mentions ADD COLUMN IF NOT EXISTS canonical_match_kind TEXT
    CHECK (canonical_match_kind IN ('representative', 'exact', 'cosine'));
ALTER TABLE relation_mentions ADD COLUMN IF NOT EXISTS canonical_similarity DOUBLE PRECISION
    CHECK (canonical_similarity >= -1 AND canonical_similarity <= 1);

CREATE TABLE IF NOT EXISTS relation_entity_links (
    relation_mention_id UUID NOT NULL
        REFERENCES relation_mentions(relation_mention_id) ON DELETE CASCADE,
    entity_mention_id UUID NOT NULL
        REFERENCES entity_mentions(entity_mention_id) ON DELETE CASCADE,
    endpoint_role TEXT NOT NULL CHECK (endpoint_role IN ('subject', 'object')),
    PRIMARY KEY (relation_mention_id, endpoint_role)
);

CREATE TABLE IF NOT EXISTS relation_ingestion_issues (
    relation_mention_id UUID NOT NULL
        REFERENCES relation_mentions(relation_mention_id) ON DELETE CASCADE,
    endpoint_role TEXT NOT NULL CHECK (endpoint_role IN ('subject', 'object')),
    issue_kind TEXT NOT NULL CHECK (issue_kind IN ('unmatched', 'ambiguous')),
    endpoint_text TEXT NOT NULL,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (relation_mention_id, endpoint_role)
);

CREATE INDEX IF NOT EXISTS idx_nodes_document_page
    ON nodes(document_code, page_number);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_node ON entity_mentions(node_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_name_lower ON entity_mentions(lower(entity_name));
CREATE INDEX IF NOT EXISTS idx_relation_mentions_node ON relation_mentions(node_id);
CREATE INDEX IF NOT EXISTS idx_relation_mentions_predicate ON relation_mentions(predicate);
CREATE INDEX IF NOT EXISTS idx_relation_entity_links_entity
    ON relation_entity_links(entity_mention_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_source_node
    ON entity_mentions(extraction_set, source_path, node_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_canonical
    ON entity_mentions(canonical_entity_id);
CREATE INDEX IF NOT EXISTS idx_relation_mentions_source_node
    ON relation_mentions(extraction_set, source_path, node_id);
CREATE INDEX IF NOT EXISTS idx_node_extraction_status_status
    ON node_extraction_status(extraction_set, status);
CREATE INDEX IF NOT EXISTS idx_relation_mentions_canonical_predicate
    ON relation_mentions(canonical_predicate_id);
CREATE INDEX IF NOT EXISTS idx_canonical_entities_type
    ON canonical_entities(entity_type);
"""


def create_schema(connection) -> None:
    """Create all tables and indexes in an already-created PostgreSQL database."""
    logger.info("Creating or updating PostgreSQL schema")
    with connection.cursor() as cursor:
        cursor.execute(SCHEMA_SQL)
    logger.info("PostgreSQL schema is ready")
