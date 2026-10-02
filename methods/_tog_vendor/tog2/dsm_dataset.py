"""Prepare the frozen DSM benchmarks for the unmodified ToG-2 pipeline.

This module only adapts dataset I/O and topic-entity preparation.  It does not
implement or alter ToG-2 search.  Every prepared ``question`` contains the
complete allowed-label set, so all downstream ToG-2 LLM prompts receive the
classification constraint together with the original question.

Supported datasets:
  * SWMH:  anxiety / bipolar disorder / depression / no mental disorders /
           suicide
  * T-SID: depression / no mental disorders / ptsd /
           suicide or self-harm tendency
  * DR:    no / yes

The entity-extraction prompt intentionally preserves the ToG-1 DSM rule: the
LLM may emit a class label as a graph entity when it judges that useful.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence


DSM_LABELS = {
    "SWMH": [
        "anxiety",
        "bipolar disorder",
        "depression",
        "no mental disorders",
        "suicide",
    ],
    "T-SID": [
        "depression",
        "no mental disorders",
        "ptsd",
        "suicide or self-harm tendency",
    ],
    "DR": ["no", "yes"],
}


TOPIC_ENTITY_EXTRACT_PROMPT = """Extract up to {k} mental-health entities from the post that can be used as starting nodes to search a DSM knowledge graph (symptoms, disorders, behaviors, substances, life events). Return one per line as {{entity}}.
If you judge it useful, you may also include any of these class labels as entities: {labels}. Only include the ones you find relevant; leave out the rest.
The final task is classification into EXACTLY ONE of these classes: {labels}.
Post and question:
{question}
A:"""


_BRACE_RE = re.compile(r"\{([^{}]+)\}")


def normalize_dataset_name(name: str) -> str:
    """Return the canonical dataset name used by ``DSM_LABELS``."""
    normalized = name.strip().upper().replace("_", "-")
    aliases = {"TSID": "T-SID", "T-SID": "T-SID", "SWMH": "SWMH", "DR": "DR"}
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported DSM dataset {name!r}; choose one of SWMH, T-SID, DR"
        )
    return aliases[normalized]


def build_classification_question(raw_question: str, labels: Sequence[str]) -> str:
    """Attach the complete answer space to the question passed through ToG-2."""
    return (
        f"{raw_question.strip()}\n"
        "Final classification instruction: choose EXACTLY ONE answer from "
        f"these allowed classes: {'; '.join(labels)}. Copy the class name exactly."
    )


def parse_entity_list(llm_output: str, limit: int = 10) -> list[str]:
    """Parse ``{entity}`` lines, with a conservative plain-line fallback."""
    braced = [match.group(1).strip() for match in _BRACE_RE.finditer(llm_output or "")]
    candidates = braced
    if not candidates:
        candidates = []
        for line in (llm_output or "").splitlines():
            value = re.sub(r"^[\d\-\.\)\*\s]+", "", line).strip().strip("{}").strip()
            if value and value.lower() not in {"none", "n/a"} and len(value.split()) <= 8:
                candidates.append(value)

    seen: set[str] = set()
    result: list[str] = []
    for value in candidates:
        key = value.casefold()
        if key and key not in seen and key != "none":
            seen.add(key)
            result.append(value)
        if len(result) >= limit:
            break
    return result


def build_entity_extraction_prompt(
    question: str, labels: Sequence[str], k: int = 10
) -> str:
    return TOPIC_ENTITY_EXTRACT_PROMPT.format(
        k=k, labels="; ".join(labels), question=question
    )


def _row_question(row: dict[str, str]) -> str:
    """Read both frozen CSV layouts, including split-column DR validation."""
    if (row.get("query") or "").strip():
        return row["query"].strip()
    post = (row.get("post") or "").strip()
    question = (row.get("question") or "").strip()
    if post or question:
        return " ".join(part for part in (post, question) if part).strip()
    raise ValueError("CSV row has neither query nor post/question text")


def load_frozen_csv(
    csv_path: str | os.PathLike[str], dataset: str, limit: int | None = None
) -> list[dict]:
    """Load one frozen split and validate every gold label."""
    dataset = normalize_dataset_name(dataset)
    labels = DSM_LABELS[dataset]
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit is not None and index >= limit:
                break
            gold = (row.get("gold_label") or "").strip()
            if gold not in labels:
                raise ValueError(
                    f"Unexpected {dataset} label {gold!r} at CSV row {index + 2}; "
                    f"allowed labels are {labels}"
                )
            raw_question = _row_question(row)
            rows.append(
                {
                    "benchmark_id": (row.get("benchmark_id") or "").strip(),
                    "source_row_index": (row.get("source_row_index") or "").strip(),
                    "dataset": dataset,
                    "raw_question": raw_question,
                    "question": build_classification_question(raw_question, labels),
                    "allowed_labels": list(labels),
                    "gold_label": gold,
                    # ToG-2 reads `answer`/`answers` as ground truth.
                    "answer": gold,
                }
            )
    return rows


def _read_cache(path: str | os.PathLike[str] | None) -> dict[str, dict]:
    cached: dict[str, dict] = {}
    if not path or not Path(path).exists():
        return cached
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cached[item["benchmark_id"]] = item
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid cache line {line_number} in {path}") from exc
    return cached


def prepare_dsm_dataset(
    csv_path: str | os.PathLike[str],
    dataset: str,
    *,
    extract_fn: Callable[[str], str] | None = None,
    link_fn: Callable[[Sequence[str]], dict[str, str]] | None = None,
    k: int = 10,
    limit: int | None = None,
    cache_path: str | os.PathLike[str] | None = None,
) -> tuple[list[dict], str]:
    """Return ToG-2 records and its question-field name.

    ``k`` caps both extracted names and linked starting entities at 10 by
    default. ``extract_fn`` receives the complete extraction prompt and returns raw LLM
    text. ``link_fn`` receives the parsed surface forms and returns the ToG-2
    topic-entity mapping ``{postgres_entity_uuid: canonical_text}``.

    Dependency injection keeps this dataset layer independent from OpenRouter,
    Milvus and PostgreSQL clients; the corresponding adapters can be supplied
    without changing this preparation logic.
    """
    records = load_frozen_csv(csv_path, dataset, limit=limit)
    cache = _read_cache(cache_path)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    for record in records:
        cached = cache.get(record["benchmark_id"])
        if cached is not None:
            if cached.get("dataset") not in (None, record["dataset"]):
                raise ValueError(
                    f"Cache entry {record['benchmark_id']} belongs to another dataset"
                )
            names = list(cached.get("extracted_entities", []))[:k]
            topic_entities = dict(list(cached.get("qid_topic_entity", {}).items())[:k])
            status = "cached"
        elif extract_fn is not None:
            prompt = build_entity_extraction_prompt(
                record["raw_question"], record["allowed_labels"], k=k
            )
            response = extract_fn(prompt)
            if not response or not response.strip():
                raise ValueError(
                    f"Empty entity-extraction response for {record['benchmark_id']}"
                )
            names = parse_entity_list(response, limit=k)
            topic_entities = (
                dict(list(link_fn(names).items())[:k])
                if names and link_fn is not None
                else {}
            )
            status = "linked" if topic_entities else ("no_link" if names else "no_entity")
            if cache_path:
                cache_item = {
                    "benchmark_id": record["benchmark_id"],
                    "dataset": record["dataset"],
                    "extracted_entities": names,
                    "qid_topic_entity": topic_entities,
                }
                # Commit each completed row so an interrupted API run resumes.
                with open(cache_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(cache_item, ensure_ascii=False) + "\n")
        else:
            names = []
            topic_entities = {}
            status = "not_run"

        record["extracted_entities"] = names
        record["entity_preparation_status"] = status
        # `qid_topic_entity` is the exact field consumed by main_tog2.py.
        record["qid_topic_entity"] = topic_entities

    return records, "question"


def openrouter_extractor(
    *, api_key: str, model: str, base_url: str, max_tokens: int = 256
) -> Callable[[str], str]:
    """Build an OpenRouter extraction callable with the loader's interface."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def extract(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract DSM graph search entities."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return extract


def existing_milvus_linker() -> Callable[[Sequence[str]], dict[str, str]]:
    """Link LLM-extracted names exactly as in ToG-1's topic preparation.

    ``ranked`` sends each name to Jina for its embedding, searches Milvus for
    nearby graph entities, and optionally prioritizes exact PostgreSQL hits.
    Jina creates vectors; Milvus performs the nearest-neighbor search.
    """
    project_root = Path(__file__).resolve().parents[2]
    tog1_source = project_root / "tog1" / "ToG" / "ToG"
    if str(tog1_source) not in sys.path:
        sys.path.insert(0, str(tog1_source))
    from milvus_linker import ranked

    def link(names: Sequence[str]) -> dict[str, str]:
        hit_lists = ranked(names, top_k=3)
        topic: dict[str, str] = {}
        seen_text: set[str] = set()

        def take(hits: Sequence[dict], allow_duplicate_text: bool) -> bool:
            for hit in hits:
                entity_id = str(hit["id"])
                entity_text = str(hit["text"])
                text_key = entity_text.strip().casefold()
                if entity_id in topic or (text_key in seen_text and not allow_duplicate_text):
                    continue
                seen_text.add(text_key)
                topic[entity_id] = entity_text
                return True
            return False

        # ToG-1 first reserves one candidate for each extracted surface form.
        for hits in hit_lists:
            take(hits, False) or take(hits, True)
        # It then adds remaining ranked candidates in round-robin order.
        for rank in range(3):
            for hits in hit_lists:
                if rank < len(hits):
                    take(hits[rank : rank + 1], False)
        return topic

    return link


def _default_csv(project_root: Path, dataset: str, split: str) -> Path:
    dataset = normalize_dataset_name(dataset)
    filename = f"{split}.csv"
    path = project_root / "dsm_grounded_v1" / dataset / filename
    if not path.exists():
        raise FileNotFoundError(f"Frozen split does not exist: {path}")
    return path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("SWMH", "T-SID", "TSID", "DR"))
    parser.add_argument("--split", default="test", choices=("test", "validation"))
    parser.add_argument("--input", help="override the frozen CSV path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--extract-entities", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    dataset = normalize_dataset_name(args.dataset)
    csv_path = Path(args.input) if args.input else _default_csv(project_root, dataset, args.split)

    extract_fn = link_fn = None
    if args.extract_entities:
        api_key = os.environ.get("LLM_API_KEY", "")
        if not api_key:
            raise ValueError("Set LLM_API_KEY when using --extract-entities")
        if not os.environ.get("EMBEDDING_API_KEY"):
            raise ValueError("Set EMBEDDING_API_KEY when using --extract-entities")
        extract_fn = openrouter_extractor(
            api_key=api_key,
            model=os.environ.get("LLM_MODEL", "google/gemma-4-31b-it"),
            base_url=os.environ.get("LLM_API_BASE", "https://openrouter.ai/api/v1"),
        )
        link_fn = existing_milvus_linker()

    records, _ = prepare_dsm_dataset(
        csv_path,
        dataset,
        extract_fn=extract_fn,
        link_fn=link_fn,
        k=args.k,
        limit=args.limit,
        cache_path=args.cache,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    linked = sum(bool(item["qid_topic_entity"]) for item in records)
    print(f"wrote {len(records)} {dataset} records to {output}; linked={linked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
