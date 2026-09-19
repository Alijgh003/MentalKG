# Pre-consolidation embeddings

This document describes the **pre-consolidation** stage only. The finalized
post-consolidation entity, predicate, triple, and source-chunk indexes are
specified in [`experiment_plan.md`](experiment_plan.md).

This phase deliberately embeds only entity mentions and the unique predicate
vocabulary. Triple embeddings are deferred until entity and predicate
canonicalisation has produced stable triples.

Collections:

- `unconsolidated_entities_v1`: one row per PostgreSQL entity mention. Exact
  repeated text is requested from the embedding API only once per run, while
  every mention keeps its node and source provenance.
- `unconsolidated_predicates_v1`: one row per unique normalized predicate,
  including its occurrence count.

Entity type is stored as filterable metadata and is not concatenated into the
embedding text. Dimensions belong to individual collection configs rather than
the embedding client: `MILVUS_ENTITY_DIMENSION` and
`MILVUS_PREDICATE_DIMENSION` are currently 128. A future chunk collection can
therefore use 512 or 1024 without changing the reusable embedder. All current
collections use cosine distance.

Configure `.env`, then run. Settings are loaded automatically through
`pydantic-settings`; sourcing `.env` in the shell is not required:

```bash
docker compose up -d milvus
.venv/bin/python -m scripts.embed_kg --kind all
```

Use `--kind entities` or `--kind predicates` independently. `--limit 100` is
available for an entity smoke test. The ingestion is resumable: existing
primary keys in Milvus are skipped before calling the embedding API.

After ingestion, verify both collections with four representative queries each:

```bash
.venv/bin/python -m scripts.smoke_test_milvus --top-k 5
```

The command prints collection row counts and ranked matches with cosine
similarity and relevant metadata. Custom queries can be supplied by repeating
`--entity-query` or `--predicate-query` up to four times.

Generate type-specific candidate clusters for consolidation:

```bash
.venv/bin/python -m scripts.cluster_milvus
```

Entity vectors are clustered independently for `symptom`, `behavior`,
`disorder`, and `concept`; predicates form their own clustering run. Clustering
uses unique normalized text and writes auditable JSONL assignments under
`outputs/clustering/`. These are candidate buckets, not automatic merge
decisions or clinical classes.
