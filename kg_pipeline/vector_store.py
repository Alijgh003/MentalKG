"""Milvus collections for pre-consolidation entity and predicate vectors."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.env import PROJECT_ENV_FILE


class MilvusConfig(BaseSettings):
    """Milvus settings loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="MILVUS_",
        env_file=PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    uri: str
    token: str = "root:Milvus"
    entity_collection: str = "unconsolidated_entities_v1"
    predicate_collection: str = "unconsolidated_predicates_v1"
    canonical_entity_collection: str = "canonical_seed_entities_v1"
    canonical_predicate_collection: str = "canonical_predicates_v1"
    triple_collection: str = "raw_triples_v1"
    source_chunk_collection: str = "source_chunks_v1"
    entity_dimension: int = Field(default=128, gt=0)
    predicate_dimension: int = Field(default=128, gt=0)
    canonical_entity_dimension: int = Field(default=128, gt=0)
    canonical_predicate_dimension: int = Field(default=128, gt=0)
    triple_dimension: int = Field(default=256, gt=0)
    source_chunk_dimension: int = Field(default=1024, gt=0)
    insert_batch_size: int = Field(default=1000, gt=0)

    @classmethod
    def from_env(cls) -> "MilvusConfig":
        """Compatibility wrapper; BaseSettings performs the actual loading."""
        return cls()


class KnowledgeGraphVectorStore:
    def __init__(self, config: MilvusConfig):
        from pymilvus import MilvusClient

        self.config = config
        self.client = MilvusClient(uri=config.uri, token=config.token)

    def ensure_collections(self) -> None:
        self._ensure_entity_collection()
        self._ensure_predicate_collection()

    def ensure_final_collections(self) -> None:
        self._ensure_canonical_entity_collection()
        self._ensure_canonical_predicate_collection()
        self._ensure_triple_collection()
        self._ensure_source_chunk_collection()

    def _index_params(self):
        params = self.client.prepare_index_params()
        params.add_index(
            field_name="vector",
            index_name="vector_hnsw",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        return params

    def _ensure_entity_collection(self) -> None:
        from pymilvus import DataType

        name = self.config.entity_collection
        if self.client.has_collection(collection_name=name):
            self._assert_dimension(name, self.config.entity_dimension)
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.config.entity_dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("normalized_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("entity_type", DataType.VARCHAR, max_length=256)
        schema.add_field("node_id", DataType.VARCHAR, max_length=128)
        schema.add_field("extraction_set", DataType.VARCHAR, max_length=64)
        schema.add_field("source_path", DataType.VARCHAR, max_length=4096)
        schema.add_field("ordinal", DataType.INT64)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=512)
        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=self._index_params(),
            consistency_level="Bounded",
        )

    def _ensure_predicate_collection(self) -> None:
        from pymilvus import DataType

        name = self.config.predicate_collection
        if self.client.has_collection(collection_name=name):
            self._assert_dimension(name, self.config.predicate_dimension)
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.config.predicate_dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=2048)
        schema.add_field("normalized_text", DataType.VARCHAR, max_length=2048)
        schema.add_field("occurrence_count", DataType.INT64)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=512)
        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=self._index_params(),
            consistency_level="Bounded",
        )

    def _ensure_canonical_entity_collection(self) -> None:
        from pymilvus import DataType

        name = self.config.canonical_entity_collection
        if self.client.has_collection(collection_name=name):
            self._assert_dimension(name, self.config.canonical_entity_dimension)
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.config.canonical_entity_dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("normalized_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("entity_type", DataType.VARCHAR, max_length=256)
        schema.add_field("canonical_name", DataType.VARCHAR, max_length=8192)
        schema.add_field("member_count", DataType.INT64)
        schema.add_field("seed_eligible", DataType.BOOL)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=512)
        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=self._index_params(),
            consistency_level="Bounded",
        )

    def _ensure_canonical_predicate_collection(self) -> None:
        from pymilvus import DataType

        name = self.config.canonical_predicate_collection
        if self.client.has_collection(collection_name=name):
            self._assert_dimension(name, self.config.canonical_predicate_dimension)
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.config.canonical_predicate_dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=2048)
        schema.add_field("normalized_text", DataType.VARCHAR, max_length=2048)
        schema.add_field("canonical_name", DataType.VARCHAR, max_length=2048)
        schema.add_field("member_count", DataType.INT64)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=512)
        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=self._index_params(),
            consistency_level="Bounded",
        )

    def _ensure_triple_collection(self) -> None:
        from pymilvus import DataType

        name = self.config.triple_collection
        if self.client.has_collection(collection_name=name):
            self._assert_dimension(name, self.config.triple_dimension)
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.config.triple_dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=16384)
        schema.add_field("subject_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("predicate", DataType.VARCHAR, max_length=2048)
        schema.add_field("object_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("node_id", DataType.VARCHAR, max_length=256)
        schema.add_field("extraction_set", DataType.VARCHAR, max_length=64)
        schema.add_field("source_path", DataType.VARCHAR, max_length=4096)
        schema.add_field("ordinal", DataType.INT64)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=512)
        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=self._index_params(),
            consistency_level="Bounded",
        )

    def _ensure_source_chunk_collection(self) -> None:
        from pymilvus import DataType

        name = self.config.source_chunk_collection
        if self.client.has_collection(collection_name=name):
            self._assert_dimension(name, self.config.source_chunk_dimension)
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.config.source_chunk_dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("node_id", DataType.VARCHAR, max_length=256)
        schema.add_field("document_code", DataType.VARCHAR, max_length=256)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("heading", DataType.VARCHAR, max_length=4096)
        schema.add_field("section_name", DataType.VARCHAR, max_length=4096)
        schema.add_field("hierarchy_path", DataType.VARCHAR, max_length=16384)
        schema.add_field("is_derived", DataType.BOOL)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=512)
        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=self._index_params(),
            consistency_level="Bounded",
        )

    def _assert_dimension(self, collection_name: str, expected_dimension: int) -> None:
        description = self.client.describe_collection(collection_name=collection_name)
        vector_field = next(field for field in description["fields"] if field["name"] == "vector")
        actual = int(vector_field["params"]["dim"])
        if actual != expected_dimension:
            raise ValueError(
                f"Collection {collection_name!r} has dimension {actual}, expected {expected_dimension}. "
                "Use a new collection name when changing embedding dimensions."
            )

    def missing_ids(self, collection_name: str, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        existing = self.client.get(collection_name=collection_name, ids=ids, output_fields=["id"])
        existing_ids = {str(record["id"]) for record in existing}
        return set(ids) - existing_ids

    def upsert(self, collection_name: str, records: list[dict]) -> None:
        if records:
            self.client.upsert(collection_name=collection_name, data=records)

    def count(self, collection_name: str) -> int:
        result = self.client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        return int(result[0]["count(*)"])

    def search(
        self,
        collection_name: str,
        vectors: list[list[float]],
        *,
        limit: int = 5,
        output_fields: tuple[str, ...] = ("text",),
    ) -> list[list[dict]]:
        """Search a cosine collection and return one ranked hit list per vector."""
        if limit <= 0:
            raise ValueError("Search limit must be positive")
        if not vectors:
            return []
        return self.client.search(
            collection_name=collection_name,
            data=vectors,
            anns_field="vector",
            limit=limit,
            output_fields=list(output_fields),
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit)}},
        )
