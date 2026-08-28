# PostgreSQL model

The file artifacts are retained exactly and also projected into queryable tables.
This deliberately avoids prematurely merging similar entity names: entity
canonicalisation is listed as future work, so an entity mention remains scoped to
the node and extraction set that produced it.

```text
documents ──< pages ──< page_parse_results
                  ^
                  │
nodes ──< node_page_spans
  │  └──< node_source_records >── ingestion_sources
  └──< derived_node_sources >── original nodes / continuation page

nodes ──< entity_mentions
nodes ──< relation_mentions ──< relation_entity_links >── entity_mentions
nodes ──< kg_source_records >── ingestion_sources
```

- `pages` keeps the original CSV markdown; `page_parse_results` retains the
  parsed page content, LLM metadata, and complete tree JSON.
- `nodes` is the normalized tree. `node_source_records` preserves each source
  version (`tree_with_IDs` and selected tree), including original JSON.
- Boundary-page records are normal `nodes` with `is_derived = true`. Their
  original/base node and next-page contribution are recorded in
  `derived_node_sources` and `node_page_spans`.
- `entity_mentions` and `relation_mentions` are extraction outputs. Their raw
  model payloads remain available, and exact in-node subject/object matches are
  represented by `relation_entity_links`.
- `kg_source_records` preserves the irregular wide main-KG records and every
  split boundary entity/relation artifact unchanged.

Run a read-only check first:

```bash
python3 -m kg_pipeline.cli --root . --validate-only
```

Create and populate PostgreSQL (after `pip install -r requirements.txt`):

```bash
python3 -m kg_pipeline.cli \
  --root . \
  --create-database \
  --admin-dsn 'postgresql://USER:PASSWORD@HOST:5432/postgres' \
  --database-name dsm5_kg \
  --dsn 'postgresql://USER:PASSWORD@HOST:5432/dsm5_kg'
```

The importer is idempotent: it upserts normalized records and replaces matching
raw source records, so rerun it after regenerating an artifact.
