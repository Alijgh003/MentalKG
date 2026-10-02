#!/usr/bin/env python3
"""Enumerate paths between seed entities up to max_hops.

Seed set = endpoints (subject_id + object_id) of 12 HippoRAG2 facts
(≈24 distinct entity IDs). Starting from each seed one by one,
traverse the minimal graph (facts as undirected edges) and whenever
another seed node is reached, store that path. Stops expanding beyond
max_hops.

Output: outputs/seed_paths/seed_paths_h<H>.json  + per-seed files
Does NOT touch hipporag code.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from pathlib import Path

import psycopg
from dotenv import load_dotenv

DEFAULT_FACT_IDS = [
    "4c360399-afa3-50df-98a3-415e8c826fa2",
    "29c000d7-3cfd-5f3e-a7ef-71af972258bf",
    "ac703ee3-0c4b-5948-943c-1df78fe36945",
    "0b46234e-0e37-58b1-8815-bcdef5161244",
    "7cc0bd9a-08ad-5d7e-a85f-3103cd69ff0b",
    "bae6cafc-42b8-5b9d-96f9-990038d34fd9",
    "042b807f-0324-5959-a1b4-dd35428dfd0e",
    "9ba4ed48-6a20-5355-9388-185e50d3034a",
    "477fe216-5bb7-53f8-bc34-56d7e0e0c74a",
    "d957e91e-92a3-508f-ae5f-70dfd0fbc0c7",
    "2f3cedd6-ccb9-5c42-8315-dfedce5d0bf5",
    "c3458b37-573d-5dfc-b2a7-022606a393e8",
]


def load_graph():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    src_dsn = os.getenv("DSM_KG_DATABASE_URL")
    if not src_dsn:
        raise SystemExit("DSM_KG_DATABASE_URL not set")
    min_db = os.getenv("DSM_MINIMAL_GRAPH_DATABASE_NAME", "dsm5_minimal_graph")
    from kg_pipeline.minimal_graph_export import target_dsn
    dsn = target_dsn(src_dsn, min_db)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id::text, text, type FROM entities")
            entities = {r[0]: {"text": r[1], "type": r[2]} for r in cur}
            cur.execute("SELECT id::text, subject_id::text, predicate, object_id::text FROM facts")
            facts = []
            # adjacency: entity_id -> list of (neighbor_id, fact_id, predicate)
            adj = defaultdict(list)
            fact_by_id = {}
            for fid, sid, pred, oid in cur:
                f = {"id": fid, "subject_id": sid, "predicate": pred, "object_id": oid}
                facts.append(f)
                fact_by_id[fid] = f
                adj[sid].append((oid, fid, pred))
                adj[oid].append((sid, fid, pred))
    return entities, adj, fact_by_id


def enumerate_paths_from_seed(start_id, seed_set, adj, max_hops, max_paths_per_seed=5000):
    """DFS limited to max_hops, cycle-free, collect paths that hit seed_set.

    Every time we step to a node that is in seed_set (and not the start at hop 0),
    we record the current path. Paths are simple (no repeated nodes).
    Returns list of dicts {nodes:[...], fact_ids:[...], predicates:[...], hops}
    """
    paths = []
    # stack: (current_node, path_nodes, path_fact_ids, path_predicates, visited_set)
    # Use iterative DFS with depth limit
    stack = [(start_id, [start_id], [], [], {start_id})]
    while stack and len(paths) < max_paths_per_seed:
        node, nodes, fact_ids, preds, visited = stack.pop()
        hops = len(fact_ids)
        if hops >= max_hops:
            continue
        for neighbor, fid, pred in adj.get(node, []):
            if neighbor in visited:
                continue
            new_nodes = nodes + [neighbor]
            new_fact_ids = fact_ids + [fid]
            new_preds = preds + [pred]
            new_hops = len(new_fact_ids)
            # if neighbor is a seed (and not trivial self-loop), record
            if neighbor in seed_set:
                # classify role: seed at start, middle, end
                # here start is seed, end is seed. Middle seeds = any interior seed
                interior_seeds = [n for n in new_nodes[1:-1] if n in seed_set]
                has_middle_seed = len(interior_seeds) > 0
                paths.append({
                    "nodes": new_nodes,
                    "fact_ids": new_fact_ids,
                    "predicates": new_preds,
                    "hops": new_hops,
                    "has_middle_seed": has_middle_seed,
                    "interior_seed_ids": interior_seeds,
                })
                if len(paths) >= max_paths_per_seed:
                    break
            # expand further if we still have budget
            if new_hops < max_hops:
                new_visited = visited | {neighbor}
                stack.append((neighbor, new_nodes, new_fact_ids, new_preds, new_visited))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=3, help="maximum hops per path")
    ap.add_argument("--max-paths-per-seed", type=int, default=5000)
    ap.add_argument("--output", type=str, default="outputs/seed_paths")
    ap.add_argument("--fact-ids", nargs="*", default=None)
    args = ap.parse_args()

    entities, adj, fact_by_id = load_graph()
    fact_ids = args.fact_ids or DEFAULT_FACT_IDS

    # derive seed entity IDs
    seed_entities = set()
    seed_fact_details = []
    for fid in fact_ids:
        f = fact_by_id.get(fid)
        if f is None:
            print(f"[WARN] fact {fid} not found")
            continue
        seed_entities.add(f["subject_id"])
        seed_entities.add(f["object_id"])
        seed_fact_details.append(f)
    seed_entities = sorted(seed_entities)
    print(f"Seed facts: {len(seed_fact_details)}, distinct seed entities: {len(seed_entities)}")
    for eid in seed_entities:
        e = entities.get(eid, {})
        print(f"  {eid}  [{e.get('type')}]  {e.get('text')}")

    seed_set = set(seed_entities)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = []
    per_seed_counts = {}

    for start_id in seed_entities:
        paths = enumerate_paths_from_seed(start_id, seed_set, adj, max_hops=args.hops, max_paths_per_seed=args.max_paths_per_seed)
        per_seed_counts[start_id] = len(paths)
        # enrich with text
        enriched = []
        for p in paths:
            enriched.append({
                **p,
                "node_texts": [entities.get(n, {}).get("text", n) for n in p["nodes"]],
                "node_types": [entities.get(n, {}).get("type") for n in p["nodes"]],
                "fact_texts": [f"{entities.get(fact_by_id[fid]['subject_id'],{}).get('text')} --{fact_by_id[fid]['predicate']}--> {entities.get(fact_by_id[fid]['object_id'],{}).get('text')}" for fid in p["fact_ids"]],
            })
        # save per-seed file
        per_path = out_dir / f"from_{start_id}_h{args.hops}.json"
        with open(per_path, "w", encoding="utf-8") as f:
            json.dump({"start_entity": {"id": start_id, **entities.get(start_id, {})}, "hops": args.hops, "paths": enriched}, f, ensure_ascii=False, indent=2)
        all_paths.extend([{"start_id": start_id, **p} for p in enriched])
        print(f"  start {start_id[:8]} -> {len(paths)} paths (h<={args.hops})")

    # sort all_paths: shortest first
    all_paths.sort(key=lambda x: (x["hops"], x["start_id"]))

    # stats
    hop_hist = defaultdict(int)
    middle_count = 0
    for p in all_paths:
        hop_hist[p["hops"]] += 1
        if p.get("has_middle_seed"):
            middle_count += 1

    summary = {
        "hops": args.hops,
        "seed_fact_ids": fact_ids,
        "seed_entity_ids": seed_entities,
        "seed_entities": [{"id": eid, **entities.get(eid, {})} for eid in seed_entities],
        "total_paths": len(all_paths),
        "per_seed_counts": per_seed_counts,
        "hop_histogram": dict(sorted(hop_hist.items())),
        "paths_with_middle_seed": middle_count,
        "max_paths_per_seed": args.max_paths_per_seed,
    }

    # save combined (without duplicating full paths if too large, but we do)
    combined_path = out_dir / f"seed_paths_h{args.hops}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "paths": all_paths}, f, ensure_ascii=False, indent=2)

    print(f"\nDone. total_paths={len(all_paths)} hist={dict(sorted(hop_hist.items()))} with_middle={middle_count}")
    print(f"Combined: {combined_path}")
    print(f"Per-seed: {out_dir}/from_<id>_h{args.hops}.json")


if __name__ == "__main__":
    main()
