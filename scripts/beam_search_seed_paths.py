#!/usr/bin/env python3
"""Beam search over seed-rooted paths.

Starts from ~21 distinct seed entities, expands via facts (undirected),
scores each path by number of distinct seed nodes in it (current simple metric),
keeps beam_size=1000 best, but guarantees at least min_per_seed paths per start seed survive.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
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
    src = os.getenv("DSM_KG_DATABASE_URL")
    assert src, "DSM_KG_DATABASE_URL missing"
    min_db = os.getenv("DSM_MINIMAL_GRAPH_DATABASE_NAME", "dsm5_minimal_graph")
    from kg_pipeline.minimal_graph_export import target_dsn
    dsn = target_dsn(src, min_db)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id::text, text, type FROM entities")
            entities = {r[0]: {"text": r[1], "type": r[2]} for r in cur}
            cur.execute("SELECT id::text, subject_id::text, predicate, object_id::text FROM facts")
            fact_by_id = {}
            adj = defaultdict(list)
            for fid, sid, pred, oid in cur:
                fact_by_id[fid] = {"id": fid, "subject_id": sid, "predicate": pred, "object_id": oid}
                adj[sid].append((oid, fid, pred))
                adj[oid].append((sid, fid, pred))
    return entities, adj, fact_by_id


def seed_entities_from_facts(fact_ids, fact_by_id):
    seeds = set()
    for fid in fact_ids:
        f = fact_by_id.get(fid)
        if f:
            seeds.add(f["subject_id"])
            seeds.add(f["object_id"])
    return sorted(seeds)


def beam_search(entities, adj, fact_by_id, seed_ids, max_hops=3, beam_size=1000, min_per_seed=2):
    seed_set = set(seed_ids)
    # initial beam: each seed as 1-node path
    # path = {start_id, nodes, fact_ids, predicates, visited, score}
    beam = []
    for sid in seed_ids:
        beam.append({
            "start_id": sid,
            "nodes": [sid],
            "fact_ids": [],
            "predicates": [],
            "visited": {sid},
            "hops": 0,
            "seed_count": 1,  # start itself is seed
            "score": 1,
        })

    # keep all paths that have at least 2 seeds (start+another) as candidates for final output
    all_valid = []  # paths where last node is seed and hops>=1 (or any path with seed_count>=2)
    seen_node_seqs: set[tuple[str, ...]] = set()  # global graph-search deduplication
    seen_fact_seqs: set[tuple[str, ...]] = set()

    for depth in range(1, max_hops + 1):
        candidates = []
        for path in beam:
            cur = path["nodes"][-1]
            for nb, fid, pred in adj.get(cur, []):
                if nb in path["visited"]:
                    continue
                new_nodes = path["nodes"] + [nb]
                new_fact_ids = path["fact_ids"] + [fid]
                # true graph-search deduplication: same node sequence (undirected) is duplicate
                node_key = tuple(new_nodes)
                rev_key = tuple(reversed(new_nodes))
                canonical_node = node_key if node_key < rev_key else rev_key
                if canonical_node in seen_node_seqs:
                    continue
                new_preds = path["predicates"] + [pred]
                new_visited = path["visited"] | {nb}
                # seed count = distinct seed nodes in new_nodes
                sc = sum(1 for n in new_nodes if n in seed_set)
                new_path = {
                    "start_id": path["start_id"],
                    "nodes": new_nodes,
                    "fact_ids": new_fact_ids,
                    "predicates": new_preds,
                    "visited": new_visited,
                    "hops": depth,
                    "seed_count": sc,
                    "score": sc,  # simple metric
                }
                candidates.append(new_path)
                seen_node_seqs.add(canonical_node)
                if sc >= 2:  # at least start + one other seed
                    all_valid.append(new_path)

        if not candidates:
            break

        # rank candidates by score desc, then hops asc (shorter preferred), then start_id
        candidates.sort(key=lambda x: (-x["score"], x["hops"], x["start_id"]))

        # enforce min_per_seed guarantee before truncating to beam_size
        # 1) pick top min_per_seed per start_id (by score)
        by_start = defaultdict(list)
        for c in candidates:
            by_start[c["start_id"]].append(c)
        guaranteed = []
        remaining = []
        for sid in seed_ids:
            lst = sorted(by_start.get(sid, []), key=lambda x: (-x["score"], x["hops"]))
            keep = lst[:min_per_seed]
            guaranteed.extend(keep)
            remaining.extend(lst[min_per_seed:])

        # 2) fill beam up to beam_size with best remaining
        remaining.sort(key=lambda x: (-x["score"], x["hops"], x["start_id"]))
        beam = guaranteed + remaining[: max(0, beam_size - len(guaranteed))]
        # re-sort beam for next iteration (best first)
        beam.sort(key=lambda x: (-x["score"], x["hops"], x["start_id"]))
        # truncate if still over (when seed_count*min_per_seed > beam_size, keep all guaranteed)
        if len(beam) > beam_size and len(guaranteed) < beam_size:
            beam = beam[:beam_size]

        print(f"depth {depth}: candidates={len(candidates)} valid_so_far={len(all_valid)} beam={len(beam)} guaranteed={len(guaranteed)}")

    # final ranking of all_valid with same criteria
    all_valid.sort(key=lambda x: (-x["score"], x["hops"], x["start_id"]))
    return all_valid, beam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--beam-size", type=int, default=1000)
    ap.add_argument("--min-per-seed", type=int, default=2)
    ap.add_argument("--output", type=str, default="outputs/beam_search")
    args = ap.parse_args()

    entities, adj, fact_by_id = load_graph()
    seed_ids = seed_entities_from_facts(DEFAULT_FACT_IDS, fact_by_id)
    print(f"Seeds: {len(seed_ids)}")
    for sid in seed_ids:
        print(f"  {sid} [{entities.get(sid,{}).get('type')}] {entities.get(sid,{}).get('text')}")

    all_valid, final_beam = beam_search(entities, adj, fact_by_id, seed_ids, max_hops=args.hops, beam_size=args.beam_size, min_per_seed=args.min_per_seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # enrich for saving
    def enrich(p):
        return {
            "start_id": p["start_id"],
            "nodes": p["nodes"],
            "node_texts": [entities.get(n, {}).get("text", n) for n in p["nodes"]],
            "node_types": [entities.get(n, {}).get("type") for n in p["nodes"]],
            "fact_ids": p["fact_ids"],
            "fact_texts": [f"{entities.get(fact_by_id[fid]['subject_id'],{}).get('text')} --{fact_by_id[fid]['predicate']}--> {entities.get(fact_by_id[fid]['object_id'],{}).get('text')}" for fid in p["fact_ids"]],
            "predicates": p["predicates"],
            "hops": p["hops"],
            "seed_count": p["seed_count"],
            "score": p["score"],
        }

    enriched_valid = [enrich(p) for p in all_valid]
    enriched_beam = [enrich(p) for p in final_beam]

    # summary
    from collections import Counter
    hist = Counter(p["hops"] for p in all_valid)
    score_hist = Counter(p["score"] for p in all_valid)
    per_seed = Counter(p["start_id"] for p in all_valid)
    summary = {
        "hops": args.hops,
        "beam_size": args.beam_size,
        "min_per_seed": args.min_per_seed,
        "seed_ids": seed_ids,
        "total_valid_paths": len(all_valid),
        "hop_histogram": dict(sorted(hist.items())),
        "score_histogram": dict(sorted(score_hist.items())),
        "per_seed_counts": {k: per_seed[k] for k in seed_ids},
        "min_per_seed_satisfied": all(per_seed.get(s,0) >= args.min_per_seed for s in seed_ids),
    }
    print(f"\nSummary: {summary}")

    with open(out_dir / f"beam_h{args.hops}_b{args.beam_size}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "paths": enriched_valid, "final_beam": enriched_beam}, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_dir / f'beam_h{args.hops}_b{args.beam_size}.json'}")


if __name__ == "__main__":
    main()
