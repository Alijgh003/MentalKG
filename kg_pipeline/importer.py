"""Idempotent PostgreSQL importer for every DSM source artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import uuid
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from .paths import DatasetPaths
from .schema import create_schema

ENTITY_NAMESPACE = uuid.UUID("dc2f570e-4c55-47c8-95af-d217a7f77e60")
RELATION_NAMESPACE = uuid.UUID("dc95d476-301a-44b1-929a-224adb2177c1")
logger = logging.getLogger(__name__)


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if line.strip():
                yield ordinal, json.loads(line)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalise_name(value: str | None) -> str:
    return (value or "").strip().casefold()


class PostgresImporter:
    """Load raw pages, trees, derived passages, entities, and triples.

    The importer is safe to run again.  Stable UUIDv5 identifiers make entity
    and relation mentions reproducible, while source-record tables retain the
    original artifacts for audits and future parser migrations.
    """

    def __init__(self, connection, paths: DatasetPaths):
        self.connection = connection
        self.paths = paths
        self._entity_by_name: dict[tuple[str, str, str], uuid.UUID] = {}

    def load(self) -> None:
        """Create the schema and load all known DSM artifacts in FK-safe order.

        Each phase commits independently.  This makes completed tables visible
        immediately to other database clients and makes a failed long-running
        import safely resumable through the existing upserts.
        """
        logger.info("Starting full DSM knowledge-graph import")
        self._run_phase("schema", lambda: create_schema(self.connection), pipelined=False)
        self._run_phase("source registration", self._register_static_sources)
        self._run_phase("raw pages", self._load_raw_pages)
        self._run_phase("parsed page results", self._load_page_parse_results)
        self._run_phase("complete tree", lambda: self._load_tree(self.paths.complete_tree, selected=False))
        self._run_phase("selected tree", lambda: self._load_tree(self.paths.selected_tree, selected=True))
        self._run_phase("derived page-boundary nodes", self._load_boundary_nodes)
        self._run_phase("rejected nodes", self._load_rejected_nodes)
        self._run_phase("main KG", self._load_main_kg)
        self._run_phase("boundary KG", self._load_boundary_kg)
        logger.info("DSM knowledge-graph import finished")

    def _run_phase(self, name: str, action, *, pipelined: bool = True) -> None:
        """Execute, flush, and commit one visible/resumable import phase."""
        logger.info("Beginning import phase: %s", name)
        if pipelined:
            # Pipeline mode removes the network round trip for every small
            # insert. It matters considerably for the 25k-node tree and the
            # thousands of separate KG artifact files.
            with self.connection.pipeline():
                action()
        else:
            action()
        self.connection.commit()
        logger.info("Completed and committed import phase: %s", name)

    @staticmethod
    def _progress(label: str, completed: int, *, interval: int = 1_000) -> None:
        if completed % interval == 0:
            logger.info("%s: %s records processed", label, completed)

    def _source_name(self, path: Path) -> str:
        return self.paths.relative_name(path)

    def _register_static_sources(self) -> None:
        logger.info("Registering source artifacts")
        for path, role in (
            (self.paths.raw_pages, "raw_pages"),
            (self.paths.parsed_pages, "page_parse_results"),
            (self.paths.complete_tree, "complete_tree"),
            (self.paths.selected_tree, "selected_tree"),
            (self.paths.boundary_nodes, "derived_nodes"),
            (self.paths.main_kg, "main_kg"),
            (self.paths.rejected_nodes, "rejected_nodes"),
        ):
            self._register_source(path, role)
        logger.info("Registered 7 static source artifacts")

    def _register_source(self, path: Path, role: str) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        source_path = self._source_name(path)
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ingestion_sources(source_path, source_role, sha256, byte_size)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_path) DO UPDATE SET
                    source_role = EXCLUDED.source_role,
                    sha256 = EXCLUDED.sha256,
                    byte_size = EXCLUDED.byte_size,
                    imported_at = now()
                """,
                (source_path, role, digest.hexdigest(), path.stat().st_size),
            )
        return source_path

    def _ensure_page(self, document_code: str, page_number: int) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO documents(document_code) VALUES (%s) ON CONFLICT DO NOTHING",
                (document_code,),
            )
            cursor.execute(
                """
                INSERT INTO pages(document_code, page_number, raw_markdown, raw_payload)
                VALUES (%s, %s, NULL, %s::jsonb)
                ON CONFLICT (document_code, page_number) DO NOTHING
                """,
                (document_code, page_number, _json({"inferred_from": "tree_or_kg"})),
            )

    def _load_raw_pages(self) -> None:
        logger.info("Loading raw pages from %s", self.paths.raw_pages)
        count = 0
        with self.paths.raw_pages.open(encoding="utf-8", newline="") as handle:
            with self.connection.cursor() as cursor:
                for batch in _batches(csv.DictReader(handle), size=250):
                    cursor.executemany(
                        "INSERT INTO documents(document_code) VALUES (%s) ON CONFLICT DO NOTHING",
                        [(document_code,) for document_code in {row["book_name"] for row in batch}],
                    )
                    cursor.executemany(
                        """
                        INSERT INTO pages(document_code, page_number, raw_markdown, raw_payload)
                        VALUES (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (document_code, page_number) DO UPDATE SET
                            raw_markdown = EXCLUDED.raw_markdown,
                            raw_payload = EXCLUDED.raw_payload
                        """,
                        [
                            (row["book_name"], int(row["page_number"]), row.get("markdown"), _json(row))
                            for row in batch
                        ],
                    )
                    count += len(batch)
                    logger.info("Raw pages: %s records queued", count)
        logger.info("Loaded %s raw pages", count)

    def _load_page_parse_results(self) -> None:
        logger.info("Loading parsed-page results from %s", self.paths.parsed_pages)
        source_path = self._source_name(self.paths.parsed_pages)
        count = 0
        for _, row in _jsonl(self.paths.parsed_pages):
            count += 1
            document_code = row["book_name"]
            page_number = int(row["page_number"])
            self._ensure_page(document_code, page_number)
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO page_parse_results(
                        document_code, page_number, parsed_content, page_tree,
                        llm_metadata, raw_payload, source_path
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                    ON CONFLICT (document_code, page_number) DO UPDATE SET
                        parsed_content = EXCLUDED.parsed_content,
                        page_tree = EXCLUDED.page_tree,
                        llm_metadata = EXCLUDED.llm_metadata,
                        raw_payload = EXCLUDED.raw_payload,
                        source_path = EXCLUDED.source_path
                    """,
                    (
                        document_code,
                        page_number,
                        row.get("content"),
                        _json(row.get("tree")),
                        _json(row.get("llm")),
                        _json(row),
                        source_path,
                    ),
                )
            self._progress("Parsed page results", count)
        logger.info("Loaded %s parsed-page results", count)

    def _load_tree(self, path: Path, *, selected: bool) -> None:
        tree_label = "selected tree" if selected else "complete tree"
        logger.info("Loading %s from %s", tree_label, path)
        source_path = self._source_name(path)
        count = 0
        for _, row in _jsonl(path):
            count += 1
            document_code = row["book_name"]
            page_number = int(row["page"])
            self._ensure_page(document_code, page_number)
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO nodes(
                        node_id, parent_node_id, document_code, page_number,
                        node_kind, tree_role, heading, content, section_name,
                        hierarchy_path, is_selected, is_derived
                    ) VALUES (
                        %(node_id)s, %(parent_id)s, %(document_code)s, %(page_number)s,
                        %(node_kind)s, %(tree_role)s, %(heading)s, %(content)s, %(section_name)s,
                        %(hierarchy_path)s, %(is_selected)s, FALSE
                    )
                    ON CONFLICT (node_id) DO UPDATE SET
                        parent_node_id = EXCLUDED.parent_node_id,
                        document_code = EXCLUDED.document_code,
                        page_number = EXCLUDED.page_number,
                        node_kind = EXCLUDED.node_kind,
                        tree_role = EXCLUDED.tree_role,
                        heading = EXCLUDED.heading,
                        content = EXCLUDED.content,
                        section_name = EXCLUDED.section_name,
                        hierarchy_path = EXCLUDED.hierarchy_path,
                        is_selected = nodes.is_selected OR EXCLUDED.is_selected
                    """,
                    {
                        "node_id": row["node_id"],
                        "parent_id": row.get("parent_id"),
                        "document_code": document_code,
                        "page_number": page_number,
                        "node_kind": row.get("type"),
                        "tree_role": row["node_type_in_tree"],
                        "heading": row.get("heading"),
                        "content": row.get("content"),
                        "section_name": row.get("section"),
                        "hierarchy_path": row.get("_path"),
                        "is_selected": selected,
                    },
                )
                cursor.execute(
                    """
                    INSERT INTO node_source_records(source_path, node_id, raw_payload)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (source_path, node_id) DO UPDATE SET raw_payload = EXCLUDED.raw_payload
                    """,
                    (source_path, row["node_id"], _json(row)),
                )
                cursor.execute(
                    """
                    INSERT INTO node_page_spans(node_id, document_code, page_number, span_role)
                    VALUES (%s, %s, %s, 'primary') ON CONFLICT DO NOTHING
                    """,
                    (row["node_id"], document_code, page_number),
                )
            self._progress(tree_label.title(), count)
        logger.info("Loaded %s %s nodes", count, tree_label)

    def _load_boundary_nodes(self) -> None:
        logger.info("Loading derived page-boundary nodes from %s", self.paths.boundary_nodes)
        source_path = self._source_name(self.paths.boundary_nodes)
        count = 0
        for _, row in _jsonl(self.paths.boundary_nodes):
            count += 1
            node_id = row["node_id"]
            original_node_id = node_id.removeprefix("with_next_page_")
            document_code = row["book_name"]
            page_number = int(row["page"])
            continuation_page = page_number + 1
            self._ensure_page(document_code, page_number)
            self._ensure_page(document_code, continuation_page)
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO nodes(
                        node_id, parent_node_id, document_code, page_number,
                        node_kind, tree_role, heading, content, section_name,
                        hierarchy_path, is_selected, is_derived
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, TRUE)
                    ON CONFLICT (node_id) DO UPDATE SET
                        parent_node_id = EXCLUDED.parent_node_id,
                        content = EXCLUDED.content,
                        section_name = EXCLUDED.section_name,
                        hierarchy_path = EXCLUDED.hierarchy_path,
                        is_derived = TRUE
                    """,
                    (
                        node_id, row.get("parent_id"), document_code, page_number,
                        row.get("type"), row["node_type_in_tree"], row.get("heading"),
                        row.get("content"), row.get("section"), row.get("_path"),
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO node_source_records(source_path, node_id, raw_payload)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (source_path, node_id) DO UPDATE SET raw_payload = EXCLUDED.raw_payload
                    """,
                    (source_path, node_id, _json(row)),
                )
                cursor.execute(
                    """
                    INSERT INTO node_page_spans(node_id, document_code, page_number, span_role)
                    VALUES (%s, %s, %s, 'primary'), (%s, %s, %s, 'continuation')
                    ON CONFLICT DO NOTHING
                    """,
                    (node_id, document_code, page_number, node_id, document_code, continuation_page),
                )
                cursor.execute(
                    """
                    INSERT INTO derived_node_sources(
                        derived_node_id, original_node_id, continuation_document_code,
                        continuation_page_number, continuation_content
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (derived_node_id) DO UPDATE SET
                        original_node_id = EXCLUDED.original_node_id,
                        continuation_document_code = EXCLUDED.continuation_document_code,
                        continuation_page_number = EXCLUDED.continuation_page_number,
                        continuation_content = EXCLUDED.continuation_content
                    """,
                    (node_id, original_node_id, document_code, continuation_page, row.get("next_page")),
                )
            self._progress("Derived page-boundary nodes", count)
        logger.info("Loaded %s derived page-boundary nodes", count)

    def _load_rejected_nodes(self) -> None:
        logger.info("Loading rejected main-KG nodes from %s", self.paths.rejected_nodes)
        source_path = self._source_name(self.paths.rejected_nodes)
        count = 0
        for _, row in _jsonl(self.paths.rejected_nodes):
            count += 1
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO node_ingestion_issues(source_path, node_id, issue_kind, raw_payload)
                    VALUES (%s, %s, 'excluded_from_main_kg', %s::jsonb)
                    ON CONFLICT (source_path, node_id, issue_kind) DO UPDATE SET
                        raw_payload = EXCLUDED.raw_payload
                    """,
                    (source_path, row["node_id"], _json(row)),
                )
        logger.info("Loaded %s rejected main-KG node records", count)

    def _load_main_kg(self) -> None:
        logger.info("Loading main KG from %s", self.paths.main_kg)
        source_path = self._source_name(self.paths.main_kg)
        for ordinal, row in _jsonl(self.paths.main_kg):
            node_id = row["node_id"]
            self._store_source_record(source_path, ordinal, node_id, "main_kg", "combined", row)
            entities = [
                value for key, value in row.items()
                if key.endswith("_entity") and isinstance(value, dict)
            ]
            relations = [
                value for key, value in row.items()
                if key.endswith("_relation") and isinstance(value, dict)
            ]
            self._store_entities("main_kg", source_path, node_id, entities)
            self._store_relations("main_kg", source_path, node_id, relations)
            self._progress("Main KG", ordinal + 1)
        logger.info("Loaded %s main-KG records", ordinal + 1)

    def _load_boundary_kg(self) -> None:
        entity_files = sorted(self.paths.boundary_kg_dir.glob("entities_with_next_page_*.json"))
        relation_files = sorted(self.paths.boundary_kg_dir.glob("relations_with_next_page_*.json"))
        logger.info(
            "Loading boundary KG from %s (%s entity files, %s relation files)",
            self.paths.boundary_kg_dir, len(entity_files), len(relation_files),
        )
        for ordinal, path in enumerate(entity_files, start=1):
            source_path = self._register_source(path, "boundary_kg_entities")
            node_id = path.stem.removeprefix("entities_")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._store_source_record(source_path, 0, node_id, "boundary_kg", "entities", payload)
            self._store_entities("boundary_kg", source_path, node_id, payload)
            self._progress("Boundary KG entity artifacts", ordinal, interval=250)
        for ordinal, path in enumerate(relation_files, start=1):
            source_path = self._register_source(path, "boundary_kg_relations")
            node_id = path.stem.removeprefix("relations_")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._store_source_record(source_path, 0, node_id, "boundary_kg", "relations", payload)
            self._store_relations("boundary_kg", source_path, node_id, payload)
            self._progress("Boundary KG relation artifacts", ordinal, interval=250)
        logger.info("Loaded %s boundary entity artifacts and %s boundary relation artifacts", len(entity_files), len(relation_files))

    def _store_source_record(
        self, source_path: str, ordinal: int, node_id: str, extraction_set: str,
        artifact_kind: str, payload: Any,
    ) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO kg_source_records(
                    source_path, record_ordinal, node_id, extraction_set, artifact_kind, raw_payload
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (source_path, record_ordinal) DO UPDATE SET
                    node_id = EXCLUDED.node_id,
                    extraction_set = EXCLUDED.extraction_set,
                    artifact_kind = EXCLUDED.artifact_kind,
                    raw_payload = EXCLUDED.raw_payload
                """,
                (source_path, ordinal, node_id, extraction_set, artifact_kind, _json(payload)),
            )

    def _store_entities(
        self, extraction_set: str, source_path: str, node_id: str, entities: list[dict[str, Any]],
    ) -> None:
        for ordinal, entity in enumerate(entities):
            name = entity.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            entity_id = uuid.uuid5(
                ENTITY_NAMESPACE, f"{extraction_set}:{source_path}:{node_id}:{ordinal}"
            )
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO entity_mentions(
                        entity_mention_id, extraction_set, node_id, source_path, ordinal,
                        entity_name, entity_type, start_char, end_char, raw_payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (entity_mention_id) DO UPDATE SET
                        entity_name = EXCLUDED.entity_name,
                        entity_type = EXCLUDED.entity_type,
                        start_char = EXCLUDED.start_char,
                        end_char = EXCLUDED.end_char,
                        raw_payload = EXCLUDED.raw_payload
                    """,
                    (
                        entity_id, extraction_set, node_id, source_path, ordinal, name,
                        entity.get("type"), entity.get("start_char"), entity.get("end_char"), _json(entity),
                    ),
                )
            self._entity_by_name.setdefault(
                (extraction_set, node_id, _normalise_name(name)), entity_id
            )

    def _store_relations(
        self, extraction_set: str, source_path: str, node_id: str, relations: list[dict[str, Any]],
    ) -> None:
        for ordinal, relation in enumerate(relations):
            subject, predicate, object_ = (
                relation.get("subject"), relation.get("predicate"), relation.get("object")
            )
            if not all(isinstance(value, str) and value.strip() for value in (subject, predicate, object_)):
                continue
            relation_id = uuid.uuid5(
                RELATION_NAMESPACE, f"{extraction_set}:{source_path}:{node_id}:{ordinal}"
            )
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO relation_mentions(
                        relation_mention_id, extraction_set, node_id, source_path, ordinal,
                        subject_text, predicate, object_text, metadata, raw_payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (relation_mention_id) DO UPDATE SET
                        subject_text = EXCLUDED.subject_text,
                        predicate = EXCLUDED.predicate,
                        object_text = EXCLUDED.object_text,
                        metadata = EXCLUDED.metadata,
                        raw_payload = EXCLUDED.raw_payload
                    """,
                    (
                        relation_id, extraction_set, node_id, source_path, ordinal,
                        subject, predicate.strip().lower(), object_, _json(relation.get("metadata")), _json(relation),
                    ),
                )
                for endpoint_role, endpoint_text in (("subject", subject), ("object", object_)):
                    entity_id = self._entity_by_name.get(
                        (extraction_set, node_id, _normalise_name(endpoint_text))
                    )
                    if entity_id:
                        cursor.execute(
                            """
                            INSERT INTO relation_entity_links(relation_mention_id, entity_mention_id, endpoint_role)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (relation_mention_id, endpoint_role) DO UPDATE SET
                                entity_mention_id = EXCLUDED.entity_mention_id
                            """,
                            (relation_id, entity_id, endpoint_role),
                        )


def create_database(admin_dsn: str, database_name: str) -> None:
    """Create a database if it does not already exist.

    PostgreSQL does not permit ``CREATE DATABASE`` inside a transaction, so this
    function deliberately uses an autocommit administrative connection.
    """
    try:
        import psycopg
        from psycopg import sql
    except ImportError as error:  # pragma: no cover - depends on local install
        raise RuntimeError("Install PostgreSQL support with: pip install -r requirements.txt") from error

    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
            if cursor.fetchone() is None:
                logger.info("Creating PostgreSQL database %s", database_name)
                cursor.execute(sql.SQL("CREATE DATABASE {} ").format(sql.Identifier(database_name)))
            else:
                logger.info("PostgreSQL database %s already exists", database_name)


def _batches(iterable, *, size: int):
    """Yield bounded lists without loading a source file into memory."""
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch
