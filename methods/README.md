# Experiment methods

Every method implements the small interface in `methods/base.py` and is exposed
through `methods/registry.py`. The common runner writes one provenance-complete
JSONL record per sample.

List registered methods and available frozen splits:

```bash
.venv/bin/python scripts/run_method.py --list
```

Run a 10-sample HippoRAG 2 retrieval smoke test (the default size):

```bash
.venv/bin/python scripts/run_method.py --method hipporag2 --dataset DR --split test
```

Run a chosen sample count or the complete split:

```bash
.venv/bin/python scripts/run_method.py --method hipporag2 --dataset SWMH --split test --limit 50
.venv/bin/python scripts/run_method.py --method hipporag2 --dataset SWMH --split test --all
```

For a reproducible non-contiguous sample (for example, a stratified set), pass
zero-based dataset row indices directly:

```bash
.venv/bin/python scripts/run_method.py \
  --method vanilla_rag --dataset SWMH --split test \
  --row-indices 23,38,87,134,148,167,245,257,308,511
```

Override method config without adding method-specific runner flags:

```bash
.venv/bin/python scripts/run_method.py \
  --method hipporag2 --dataset DR --split validation --limit 20 \
  --method-option recognition_mode=llm \
  --method-option fact_retrieval_top_k_per_query=15 \
  --method-option fact_filter_candidate_limit=20 \
  --method-option final_fact_top_k=10 \
  --method-option passage_similarity_threshold=0.50 \
  --method-option passage_seed_mass_ratio=0.10 \
  --method-option retrieval_top_k=20
```

`recognition_mode=llm` is the default. DSPy first converts the psychiatric post
into up to `fact_query_limit` subject-predicate-object search triples. Every
generated triple is embedded and searched independently in the raw-triple
Milvus collection. By default, each query retrieves 15 candidates. Duplicate
fact IDs are fused by their best cosine score, the strongest hit from each
query is preserved when possible, and the remaining slots are filled globally
by cosine similarity. One DSPy call filters only the resulting top 20 candidates
across all queries and returns at most 10 facts for graph seeding. Use
`recognition_mode=embedding` only as an ablation that searches once with the
raw post and makes no LLM call.

Dense passage search still examines up to 200 candidates, but only candidates
whose raw cosine similarity is at least `passage_similarity_threshold` (default
`0.50`) can seed PPR. Their combined restart mass is capped at
`passage_seed_mass_ratio` (default `0.10`) times the entity-seed mass, preventing
a large dense candidate pool from overwhelming the graph-derived seeds.

## Vanilla RAG baseline

`vanilla_rag` embeds the full input, retrieves the 50 nearest chunks from
Milvus, asks a reranking endpoint to retain the best 10, and sends the best 5
to the same dataset-aware DSPy answer program used by HippoRAG:

```bash
.venv/bin/python scripts/run_method.py \
  --method vanilla_rag --dataset DR --split test --limit 10 \
  --stop-after answer_generation
```

The sizes are configurable with `VANILLA_RAG_RETRIEVAL_TOP_K`,
`VANILLA_RAG_RERANK_TOP_K`, and `VANILLA_RAG_QA_TOP_K`. Reranker connection
settings use `RERANK_BASE_URL`, `RERANK_API_KEY`, and `RERANK_MODEL`; when they
are omitted, they inherit the existing embedding service settings.

The currently configured local service advertises `/v1/rerank` and accepts the
same `jina` model loaded for embeddings. In a smoke test its rerank scores and
ordering were effectively identical to dense cosine retrieval, so it should
not be treated as a distinct cross-encoder. To measure a real reranking gain,
point the `RERANK_*` variables at a service running a dedicated reranker model.

## Inspect one pipeline boundary

Use `--offset` and `--limit 1` to select one dataset row, and `--stop-after` to
execute only through the desired boundary. The selected stage output is printed
as JSON and is also stored under `stage_outputs` in the main run JSONL.

```bash
# LLM-generated search triples only
.venv/bin/python scripts/run_method.py \
  --method hipporag2 --dataset DR --split test --offset 0 --limit 1 \
  --stop-after fact_generation

# Retrieved graph facts, including text, scores, and the generated query that matched
.venv/bin/python scripts/run_method.py \
  --method hipporag2 --dataset DR --split test --offset 0 --limit 1 \
  --stop-after fact_retrieval

# Dense chunk candidates / entity reset weights / raw PPR node scores / final chunks
.venv/bin/python scripts/run_method.py --method hipporag2 --dataset DR --limit 1 --stop-after passage_retrieval
.venv/bin/python scripts/run_method.py --method hipporag2 --dataset DR --limit 1 --stop-after seed_weighting
.venv/bin/python scripts/run_method.py --method hipporag2 --dataset DR --limit 1 --stop-after ppr
.venv/bin/python scripts/run_method.py --method hipporag2 --dataset DR --limit 1 --stop-after passage_ranking
```

The supported boundaries are `fact_generation`, `fact_retrieval`,
`passage_retrieval`, `seed_weighting`, `ppr`, and `passage_ranking`.
The `passage_retrieval` terminal preview shows only its top 10 chunks for
readability; the run JSONL still stores the complete candidate list.

Add `--generate-answer` to send the top `qa_top_k` final passages (default 5)
to the shared DSPy language model and store the answer, explanation, and cited
passage IDs. `--stop-after answer_generation` enables the same final stage and
prints only its structured output.

Generated run artifacts are written under `outputs/method_runs/` by default.
They intentionally contain the immutable input and retrieved evidence required
by `KG-Construction/evaluation_protocol.md`; do not publish restricted dataset
artifacts.

## Explanation evaluation

The first explanation metric is an embedding-based semantic similarity score.
The evaluator extracts the generated reasoning and the benchmark's reference
reasoning, embeds both with the configured embedding service, and reports their
cosine similarity per sample plus mean and median:

```bash
.venv/bin/python scripts/evaluate_explanations.py \
  --run outputs/method_runs/vanilla_rag-SWMH-test-stratified-10-final.jsonl \
  --dataset SWMH --split test \
  --output outputs/method_runs/vanilla_rag-SWMH-explanation-score.json
```

The result is named `bert_score_cosine` for experiment tracking, but is a
whole-explanation BERTScore-style proxy, not the original token-level BERTScore
implementation. The reference field is detected from `response` or
`gpt-3.5-turbo`; use `--reference-field` to override it.

Each run produces three synchronized files:

- `<run>.jsonl`: one complete provenance and result record per sample;
- `<run>.events.jsonl`: one structured event per startup or sample stage;
- `<run>.log`: human-readable live progress, warnings, retries, and tracebacks.

Every sample event carries `dataset`, `split`, zero-based `dataset_row_index`,
original `source_row_index`, and stable `sample_id`. Stage events record status,
latency, LLM/embedding/retrieval call counts, input/output LLM tokens when the
provider reports them, and stage-specific details. Files are flushed after every
sample, so completed work remains inspectable if a later sample fails.

## Local run viewer

For a readable view of the JSONL artifacts, start the dependency-free local UI:

```bash
.venv/bin/python scripts/serve_method_viewer.py
```

Then open `http://127.0.0.1:8765`. The viewer automatically lists run files in
`outputs/method_runs`, keeps all data local, and presents samples, generated
triples, retrieved facts, raw/normalized scores, matched queries, PPR nodes, and
ranked chunks as navigable cards and tables.
