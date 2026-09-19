"""Locations of the versioned, file-based DSM data set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    """All inputs required to reproduce the PostgreSQL data set.

    Paths are intentionally explicit rather than scattered through notebooks and
    task modules.  A different book collection can be loaded by constructing
    this object with another root directory or replacing individual paths.
    """

    root: Path

    @property
    def data_root(self) -> Path:
        """Return either a repository's ``books`` directory or a direct data directory."""
        nested = self.root / "books"
        return nested if nested.is_dir() else self.root

    @property
    def raw_pages(self) -> Path:
        return self.data_root / "dsm5_all.csv"

    @property
    def parsed_pages(self) -> Path:
        return self.data_root / "dsm-tree/dsm5-final.jsonl"

    @property
    def complete_tree(self) -> Path:
        return self.data_root / "dsm-tree/tree_with_IDs.json"

    @property
    def selected_tree(self) -> Path:
        return self.data_root / "dsm-tree/dsm5_selected_chapters_tree.jsonl"

    @property
    def boundary_nodes(self) -> Path:
        return self.data_root / "dsm-tree/last_concated_with_first_nodes.jsonl"

    @property
    def main_kg(self) -> Path:
        return self.data_root / "dsm5-KG/dsm5-KG.jsonl"

    @property
    def rejected_nodes(self) -> Path:
        return self.data_root / "dsm5-KG/dsm5-bad_nodes.jsonl"

    @property
    def boundary_kg_dir(self) -> Path:
        return self.data_root / "dsm5-KG/last_first_kg/dsm5-KG"

    def relative_name(self, path: Path) -> str:
        """Return a stable source identifier stored in the database."""
        return path.resolve().relative_to(self.data_root.resolve()).as_posix()

    def required_files(self) -> tuple[Path, ...]:
        return (
            self.raw_pages,
            self.parsed_pages,
            self.complete_tree,
            self.selected_tree,
            self.boundary_nodes,
            self.main_kg,
            self.rejected_nodes,
        )
