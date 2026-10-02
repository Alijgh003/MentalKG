"""Wrapper: SWMH/DSM dataset -> ToG-compatible format.

SWMH has no `topic_entity` (ToG beam-search start nodes), so without this
`main_freebase.py` falls back to plain CoT and the KG is never used.
This wrapper:
  1. reads `query` from dsm_grounded_v1 CSVs,
  2. extracts candidate entity surface forms with the SAME LLM used by ToG,
  3. links them to Postgres KG ids (stub now, real SQL later),
  4. returns (datas, question_string) exactly like utils.prepare_dataset().

Usage as library:
    from dsm_wrapper import prepare_dsm_dataset
    datas, question_string = prepare_dsm_dataset(csv_path, args, link_fn=my_pg_linker)

Usage as CLI (caches LLM output so you pay once):
    python dsm_wrapper.py --input ../../dsm_grounded_v1/SWMH/test.csv \
        --output dsm_swmh_tog.json --LLM_type gpt-3.5-turbo --opeani_api_keys sk-...

Once Postgres schema arrives, only replace `postgres_link_names()`.
"""
import argparse
import csv
import json
import os
import re


TOPIC_ENTITY_EXTRACT_PROMPT = """Extract mental-health entities from the post that can be used as starting nodes to search a DSM knowledge graph (symptoms, disorders, behaviors, substances, life events). Return one per line as {{entity}}.
If you judge it useful, you may also include any of these class labels as entities: {labels}. Only include the ones you find relevant, leave out the rest.
Note: the final task is classification into ONE of these classes: {labels}.
Post: {post}
Question: {question}
A:"""

# Allowed answer classes per dataset (from dsm_grounded_v1/manifest.json
# selected_label_counts + verified against CSVs). SWMH=5, DR=2, T-SID=4.
SWMH_LABELS = ["anxiety", "bipolar disorder", "depression",
               "no mental disorders", "suicide"]
DR_LABELS = ["no", "yes"]
TSID_LABELS = ["depression", "no mental disorders", "ptsd",
               "suicide or self-harm tendency"]
DSM_LABELS = {"SWMH": SWMH_LABELS, "DR": DR_LABELS, "T-SID": TSID_LABELS}

DSM_ANSWER_PROMPT_TEMPLATE = """Given a post and the retrieved DSM knowledge graph triplets (entity, relation, entity), classify the post into EXACTLY ONE of these classes: {labels}.
Post: {question}
Knowledge Triplets: {triplets}
First write 2-4 sentences of human explanation that reasons ONLY from the triplets above: name the triplets you use and how they point to the label. Then, on the last line, return ONLY one of the classes above wrapped in double braces, e.g. {{{{depression}}}}. Copy the class name exactly; never put entity names or (via ...) notes in the answer line.
A: """


def build_dsm_answer_prompt(question, triplets_str, labels=SWMH_LABELS):
    return DSM_ANSWER_PROMPT_TEMPLATE.format(
        labels="; ".join(labels), question=question, triplets=triplets_str)

# matches {entity} or 1. entity / - entity lines
_BRACE_RE = re.compile(r"\{([^{}]+)\}")


def load_swmh_csv(path, limit=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if limit is not None and i >= limit:
                break
            rows.append({
                "benchmark_id": row.get("benchmark_id", ""),
                "question": row.get("query", ""),
                "gold_label": row.get("gold_label", ""),
                "source_row_index": row.get("source_row_index", ""),
            })
    return rows


def parse_entity_list(llm_output):
    """Robust parse: prefers {x}, falls back to newline/comma split."""
    cands = [m.group(1).strip() for m in _BRACE_RE.finditer(llm_output or "")]
    if cands:
        return dedup([c for c in cands if c and c.lower() != "none"])[:10]
    out = []
    for line in (llm_output or "").splitlines():
        line = re.sub(r"^[\d\-\.\)\*\s]+", "", line).strip().strip("{}").strip()
        if not line or line.lower() in ("none", "n/a"):
            continue
        # avoid swallowing full sentences
        if len(line.split()) > 6:
            continue
        out.append(line)
    if not out and llm_output and len(llm_output.split()) <= 6:
        out = [llm_output.strip()]
    return dedup(out)[:10]


def dedup(names):
    seen, out = set(), []
    for n in names:
        k = n.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(n.strip())
    return out


def extract_names_llm(question, run_llm_fn, k=10, temperature=0.0, max_tokens=256,
                       llm_type="gpt-3.5-turbo", api_keys="", labels=""):
    # post may already contain "Question:" suffix; keep whole text, LLM handles it
    prompt = TOPIC_ENTITY_EXTRACT_PROMPT.format(k=k, post=question[:3000],
                                                question=question[-500:],
                                                labels=labels)
    out = run_llm_fn(prompt, temperature, max_tokens, api_keys, llm_type)
    return parse_entity_list(out)


def postgres_link_names(names, link_fn=None):
    """Map surface names -> ToG topic_entity {id: display_name}.

    link_fn: callable(name) -> (kg_id, canonical_name) | None.
    Replace with real Postgres lookup when schema arrives, e.g.:
        def pg_link(name):
            cur.execute("SELECT id, name FROM entities WHERE name ILIKE %s LIMIT 1", (name,))
            ...
    Until then: identity mapping {name: name} so ToG can start searching
    (keys will be swapped for real ids later without changing callers).
    """
    topic = {}
    for n in names:
        if link_fn is not None:
            hit = link_fn(n)
            if hit is None:
                continue
            kg_id, canon = hit
            topic[str(kg_id)] = canon
        else:
            topic[n] = n
    return topic


def prepare_dsm_dataset(csv_path, run_llm_fn=None, link_fn=None, k=5,
                        llm_type="gpt-3.5-turbo", api_keys="", limit=None,
                        cache_path=None):
    """Drop-in replacement for utils.prepare_dataset(). Returns (datas, 'question')."""
    rows = load_swmh_csv(csv_path, limit=limit)
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    cache[d["question"]] = d["topic_entity"]
                except Exception:
                    pass
    datas = []
    for row in rows:
        q = row["question"]
        if q in cache:
            topic = cache[q]
        elif run_llm_fn is None:
            topic = {}  # no LLM: ToG will CoT-fallback; linker fills later
        else:
            names = extract_names_llm(q, run_llm_fn, k=k,
                                      llm_type=llm_type, api_keys=api_keys)
            topic = postgres_link_names(names, link_fn=link_fn)
        datas.append({
            "question": q,
            "topic_entity": topic,       # freebase-style path
            "qid_topic_entity": topic,   # wiki-style path (same until PG ids)
            "gold_label": row["gold_label"],
            "benchmark_id": row["benchmark_id"],
        })
    if cache_path and run_llm_fn is not None:
        with open(cache_path, "w", encoding="utf-8") as f:
            for d in datas:
                f.write(json.dumps({"question": d["question"],
                                    "topic_entity": d["topic_entity"]}) + "\n")
    return datas, "question"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="dsm_swmh_tog.json")
    ap.add_argument("--cache", default="dsm_topic_cache.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true",
                    help="only convert format, skip entity extraction")
    ap.add_argument("--LLM_type", default="gpt-3.5-turbo")
    ap.add_argument("--opeani_api_keys", default="")
    ap.add_argument("--max_length", type=int, default=256)
    a = ap.parse_args()

    run_fn = None
    if not a.no_llm:
        from utils import run_llm as _run
        run_fn = _run

    datas, _ = prepare_dsm_dataset(
        a.input, run_llm_fn=run_fn, k=a.k, llm_type=a.LLM_type,
        api_keys=a.opeani_api_keys, limit=a.limit, cache_path=a.cache)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=1)
    linked = sum(1 for d in datas if d["topic_entity"])
    print(f"wrote {len(datas)} records -> {a.output} | with entities: {linked}")
    print("wire into ToG: datas, qs = prepare_dsm_dataset(csv, run_llm_fn=run_llm, ...)")
