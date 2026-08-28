"""Read-only validation of the file artifacts before a database import."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .paths import DatasetPaths


@dataclass(frozen=True)
class ValidationReport:
    raw_pages: int
    parsed_pages: int
    complete_nodes: int
    selected_nodes: int
    selected_leaf_nodes: int
    boundary_nodes: int
    main_kg_records: int
    rejected_nodes: int
    boundary_entity_files: int
    boundary_relation_files: int
    missing_main_kg_nodes: int
    rejected_missing_main_kg_nodes: int


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if line.strip():
                yield ordinal, json.loads(line)


def validate_dataset(paths: DatasetPaths) -> ValidationReport:
    """Validate cross-file node IDs without requiring PostgreSQL or ML packages."""
    missing = [path for path in paths.required_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required source files are missing: " + ", ".join(map(str, missing)))

    with paths.raw_pages.open(encoding="utf-8", newline="") as handle:
        raw_pages = sum(1 for _ in csv.DictReader(handle))

    parsed_pages = sum(1 for _ in _jsonl(paths.parsed_pages))
    complete_nodes = [record for _, record in _jsonl(paths.complete_tree)]
    selected_nodes = [record for _, record in _jsonl(paths.selected_tree)]
    boundary_nodes = [record for _, record in _jsonl(paths.boundary_nodes)]
    main_kg = [record for _, record in _jsonl(paths.main_kg)]
    rejected = [record for _, record in _jsonl(paths.rejected_nodes)]

    selected_ids = {record["node_id"] for record in selected_nodes}
    leaf_ids = {
        record["node_id"]
        for record in selected_nodes
        if record["node_type_in_tree"] == "leaf"
    }
    main_ids = {record["node_id"] for record in main_kg}
    rejected_ids = {record["node_id"] for record in rejected}
    unknown_main_ids = main_ids - selected_ids
    if unknown_main_ids:
        raise ValueError(f"Main KG refers to {len(unknown_main_ids)} unknown selected-tree node IDs")
    unknown_boundary_bases = {
        record["node_id"].removeprefix("with_next_page_")
        for record in boundary_nodes
    } - selected_ids
    if unknown_boundary_bases:
        raise ValueError(f"Boundary nodes refer to {len(unknown_boundary_bases)} unknown base node IDs")

    boundary_files = list(paths.boundary_kg_dir.glob("*.json"))
    file_kinds = Counter(
        "entities" if path.name.startswith("entities_with_next_page_") else "relations"
        for path in boundary_files
    )
    return ValidationReport(
        raw_pages=raw_pages,
        parsed_pages=parsed_pages,
        complete_nodes=len(complete_nodes),
        selected_nodes=len(selected_nodes),
        selected_leaf_nodes=len(leaf_ids),
        boundary_nodes=len(boundary_nodes),
        main_kg_records=len(main_kg),
        rejected_nodes=len(rejected_ids),
        boundary_entity_files=file_kinds["entities"],
        boundary_relation_files=file_kinds["relations"],
        missing_main_kg_nodes=len(leaf_ids - main_ids),
        rejected_missing_main_kg_nodes=len((leaf_ids - main_ids) & rejected_ids),
    )
