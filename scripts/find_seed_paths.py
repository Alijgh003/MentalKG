#!/usr/bin/env python3
"""Find short simple paths whose endpoints are retrieved-fact seed entities."""
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

DEFAULT_FACT_IDS = [
    "4c360399-afa3-50df-98a3-415e8c826fa2", "29c000d7-3cfd-5f3e-a7ef-71af972258bf",
    "ac703ee3-0c4b-5948-943c-1df78fe36945", "0b46234e-0e37-58b1-8815-bcdef5161244",
    "7cc0bd9a-08ad-5d7e-a85f-3103cd69ff0b", "bae6cafc-42b8-5b9d-96f9-990038d34fd9",
    "042b807f-0324-5959-a1b4-dd35428dfd0e", "9ba4ed48-6a20-5355-9388-185e50d3034a",
    "477fe216-5bb7-53f8-bc34-56d7e0e0c74a", "d957e91e-92a3-508f-ae5f-70dfd0fbc0c7",
    "2f3cedd6-ccb9-5c42-8315-dfedce5d0bf5", "c3458b37-573d-5dfc-b2a7-022606a393e8",
]


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


def find_paths(seed_ids, facts, adjacency, max_hops):
    seed_set = set(seed_ids)
    paths = {}

    def dfs(start, current, nodes, fact_ids, predicates, directions, visited):
        if len(fact_ids) >= max_hops:
            return
        for neighbor, fact_id, predicate, direction in adjacency.get(current, ()):
            if neighbor in visited:
                continue
            # Every intermediate entity must be object in one fact and subject
            # in the next. This is equivalent to keeping one traversal direction
            # for the complete path (all forward or all backward).
            if directions and direction != directions[-1]:
                continue
            next_nodes = nodes + [neighbor]
            next_fact_ids = fact_ids + [fact_id]
            next_predicates = predicates + [predicate]
            next_directions = directions + [direction]
            if neighbor in seed_set and neighbor != start:
                forward = tuple(next_nodes)
                reverse = tuple(reversed(next_nodes))
                key = min(forward, reverse)
                paths.setdefault(key, {
                    "start_id": start,
                    "end_id": neighbor,
                    "nodes": next_nodes,
                    "fact_ids": next_fact_ids,
                    "predicates": next_predicates,
                    "traversal_directions": next_directions,
                    "hops": len(next_fact_ids),
                })
                continue
            dfs(start, neighbor, next_nodes, next_fact_ids, next_predicates, next_directions, visited | {neighbor})

    for seed in seed_ids:
        dfs(seed, seed, [seed], [], [], [], {seed})
    return sorted(paths.values(), key=lambda path: (path["hops"], path["nodes"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--output", default="outputs/seed_paths/seed_to_seed_dfs_h3.json")
    parser.add_argument("--fact-ids", nargs="*", default=DEFAULT_FACT_IDS)
    args = parser.parse_args()

    entities, facts, adjacency = load_graph()
    seed_entities = sorted({
        endpoint
        for fact_id in args.fact_ids
        if fact_id in facts
        for endpoint in (facts[fact_id]["subject_id"], facts[fact_id]["object_id"])
    })
    paths = find_paths(seed_entities, facts, adjacency, args.hops)
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fact_ids": args.fact_ids,
        "seed_entities": [{"id": eid, **entities[eid]} for eid in seed_entities],
        "max_hops": args.hops,
        "summary": {"seed_count": len(seed_entities), "path_count": len(paths),
                    "paths_by_hop": {str(h): sum(p["hops"] == h for p in paths) for h in range(1, args.hops + 1)}},
        "paths": paths,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
