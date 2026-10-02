#!/usr/bin/env python3
"""Find short simple graph paths from model-query seeds to label-query seeds.

Usage:
  .venv/bin/python scripts/find_query_to_label_paths.py \
    --run outputs/method_runs/<fact-retrieval-run>.jsonl --hops 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kg_pipeline.minimal_graph_export import target_dsn


def load_graph():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    source = os.getenv("DSM_KG_DATABASE_URL")
    if not source:
        raise SystemExit("DSM_KG_DATABASE_URL is not set")
    database = os.getenv("DSM_MINIMAL_GRAPH_DATABASE_NAME", "dsm5_minimal_graph")
    with psycopg.connect(target_dsn(source, database)) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id::text, text, type FROM entities")
            entities = {row[0]: {"text": row[1], "type": row[2]} for row in cur}
            cur.execute("SELECT id::text, subject_id::text, predicate, object_id::text FROM facts")
            facts = {}
            adjacency = defaultdict(list)
            for fid, subject, predicate, object_ in cur:
                facts[fid] = {"id": fid, "subject_id": subject, "predicate": predicate, "object_id": object_}
                adjacency[subject].append((object_, fid, predicate, "subject_to_object"))
                adjacency[object_].append((subject, fid, predicate, "object_to_subject"))
    return entities, facts, adjacency


def classify_fact_ids(record: dict) -> tuple[set[str], set[str]]:
    """Classify retrieved fact IDs using matched query provenance."""
    query_seeds: set[str] = set()
    label_seeds: set[str] = set()
    for fact in record.get("stage_outputs", {}).get("fact_retrieval", []):
        kinds = {
            str(match.get("query", {}).get("query_type", "model_generated"))
            for match in fact.get("matched_queries", [])
        }
        if "label_target" in kinds:
            label_seeds.add(str(fact["triple_id"]))
        if "model_generated" in kinds or not kinds:
            query_seeds.add(str(fact["triple_id"]))
    return query_seeds, label_seeds


def endpoint_seeds(fact_ids: set[str], facts: dict[str, dict]) -> set[str]:
    return {
        endpoint
        for fact_id in fact_ids
        if fact_id in facts
        for endpoint in (facts[fact_id]["subject_id"], facts[fact_id]["object_id"])
    }


def find_paths(starts: set[str], targets: set[str], adjacency, max_hops: int):
    paths: dict[tuple[str, ...], dict] = {}

    def dfs(start, current, nodes, fact_ids, predicates, directions, visited):
        if len(fact_ids) >= max_hops:
            return
        for neighbor, fact_id, predicate, direction in adjacency.get(current, ()):
            if neighbor in visited:
                continue
            # A valid bridge is object in one relation and subject in the next;
            # direction therefore cannot switch inside a path.
            if directions and direction != directions[-1]:
                continue
            next_nodes = nodes + [neighbor]
            next_fact_ids = fact_ids + [fact_id]
            next_predicates = predicates + [predicate]
            next_directions = directions + [direction]
            if neighbor in targets and neighbor != start:
                key = min(tuple(next_nodes), tuple(reversed(next_nodes)))
                paths.setdefault(key, {
                    "start_entity_id": start,
                    "end_entity_id": neighbor,
                    "nodes": next_nodes,
                    "fact_ids": next_fact_ids,
                    "predicates": next_predicates,
                    "traversal_directions": next_directions,
                    "hops": len(next_fact_ids),
                })
                continue
            dfs(start, neighbor, next_nodes, next_fact_ids, next_predicates, next_directions, visited | {neighbor})

    for start in starts:
        dfs(start, start, [start], [], [], [], {start})
    return sorted(paths.values(), key=lambda path: (path["hops"], path["nodes"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="fact-retrieval JSONL run")
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.run).read_text(encoding="utf-8").splitlines() if line.strip()]
    record = records[args.sample_index]
    entities, facts, adjacency = load_graph()
    query_fact_ids, label_fact_ids = classify_fact_ids(record)
    query_entities = endpoint_seeds(query_fact_ids, facts)
    label_entities = endpoint_seeds(label_fact_ids, facts)
    paths = find_paths(query_entities, label_entities, adjacency, args.hops)
    for path in paths:
        path["node_texts"] = [entities.get(node, {}).get("text", node) for node in path["nodes"]]
        path["node_types"] = [entities.get(node, {}).get("type") for node in path["nodes"]]
        path["fact_texts"] = [
            f"{entities[facts[fid]['subject_id']]['text']} --{facts[fid]['predicate']}--> "
            f"{entities[facts[fid]['object_id']]['text']}"
            for fid in path["fact_ids"]
        ]
        path["steps"] = [
            {
                "from_entity": path["node_texts"][index],
                "to_entity": path["node_texts"][index + 1],
                "traversal_direction": path["traversal_directions"][index],
                "fact_direction": path["fact_texts"][index],
            }
            for index in range(path["hops"])
        ]
    output = Path(args.output or f"outputs/seed_paths/query_to_label_{record['sample_id']}_h{args.hops}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "model_generated_fact_endpoints",
        "destination": "label_target_fact_endpoints",
        "run": args.run,
        "sample_id": record.get("sample_id"),
        "max_hops": args.hops,
        "query_fact_ids": sorted(query_fact_ids),
        "label_fact_ids": sorted(label_fact_ids),
        "query_seed_entities": [{"id": eid, **entities[eid]} for eid in sorted(query_entities)],
        "label_seed_entities": [{"id": eid, **entities[eid]} for eid in sorted(label_entities)],
        "summary": {"query_seed_count": len(query_entities), "label_seed_count": len(label_entities),
                    "path_count": len(paths),
                    "paths_by_hop": {str(h): sum(p["hops"] == h for p in paths) for h in range(1, args.hops + 1)}},
        "paths": paths,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
