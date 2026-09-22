# Minimal four-table graph database

The export database contains exactly four application tables:

- `chunks(id, content)`: selected leaf nodes only; content is stored first and
  document hierarchy is appended after a blank line;
- `entities(id, text, type)`: canonical rows for consolidated types, individual
  mention rows for non-consolidated types, and individual `unresolved` rows for
  relation endpoints without a definite entity link;
- `facts(id, subject_id, predicate, object_id)`: one row per unique canonical
  triple; the ID is stable for the triple;
- `mentions(fact_id, chunk_id)`: many-to-many links between unique facts and
  every selected leaf source in which the fact was mentioned.

Duplicate relation mentions are collapsed by `(subject_id, predicate, object_id)`.
No labels, raw payloads, embeddings, scores, or duplicate provenance columns are
copied. Canonical entity IDs are reused from the source database; fact IDs are
derived deterministically from the canonical triple.

```bash
.venv/bin/python -m scripts.build_minimal_graph_db
```

The default database name is `dsm5_minimal_graph`; override it with
`--database-name` or `DSM_MINIMAL_GRAPH_DATABASE_NAME`.
