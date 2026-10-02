#!/usr/bin/env python3
"""Build 5-hop subgraphs for 12 HippoRAG2 fact IDs without touching hipporag code.

Uses the minimal retrieval DB (dsm5_minimal_graph) only.
Saves 12 JSON files under outputs/fact_subgraphs/<fact_id>.json

Usage:
  .venv/bin/python scripts/build_fact_subgraphs.py
  .venv/bin/python scripts/build_fact_subgraphs.py --hops 5 --output outputs/fact_subgraphs
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque, defaultdict
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from kg_pipeline.minimal_graph_export import target_dsn

# 12 fact IDs from hipporag2-SWMH-one-end-to-end-6plus6.jsonl
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


def load_graph(conn):
    """Load entities, facts into memory for BFS."""
    with conn.cursor() as cur:
        cur.execute("SELECT id::text, text, type FROM entities")
        entities = {row[0]: {"text": row[1], "type": row[2]} for row in cur}
        cur.execute("SELECT id::text, subject_id::text, predicate, object_id::text FROM facts")
        facts = []
        adj = defaultdict(list)  # entity_id -> list of fact indices
        for fid, sid, pred, oid in cur:
            idx = len(facts)
            facts.append({"id": fid, "subject_id": sid, "predicate": pred, "object_id": oid})
            adj[sid].append(idx)
            adj[oid].append(idx)
        cur.execute("SELECT fact_id::text, chunk_id::text FROM mentions")
        fact_chunks = defaultdict(list)
        chunk_facts = defaultdict(list)
        for fid, cid in cur:
            fact_chunks[fid].append(cid)
            chunk_facts[cid].append(fid)
        cur.execute("SELECT id, content FROM chunks")
        chunks = {row[0]: row[1] for row in cur}
    return entities, facts, adj, fact_chunks, chunk_facts, chunks


def bfs_subgraph(start_entities, facts, adj, max_hops=5):
    """BFS up to max_hops factual edges (undirected)."""
    visited_entities = set(start_entities)
    visited_fact_ids = set()
    # distance per entity (hops from any start)
    dist = {e: 0 for e in start_entities}
    queue = deque(start_entities)
    # track how each entity was reached (for debugging / edge trace)
    # BFS layer by layer: each pop explores its incident facts -> neighbor entities
    while queue:
        entity = queue.popleft()
        d = dist[entity]
        if d >= max_hops:
            continue
        for fact_idx in adj.get(entity, []):
            fact = facts[fact_idx]
            fid = fact["id"]
            neighbor = fact["object_id"] if fact["subject_id"] == entity else fact["subject_id"]
            # include fact even if neighbor already visited (it still belongs to subgraph)
            if fid not in visited_fact_ids:
                visited_fact_ids.add(fid)
            if neighbor not in visited_entities:
                visited_entities.add(neighbor)
                dist[neighbor] = d + 1
                queue.append(neighbor)
            else:
                # if neighbor already visited at larger distance, no update needed
                # but fact is already included
                pass
    # collect fact objects
    fact_map = {f["id"]: f for f in facts}
    subgraph_facts = [fact_map[fid] for fid in visited_fact_ids]
    return visited_entities, subgraph_facts, dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hops", type=int, default=5)
    parser.add_argument("--output", type=str, default="outputs/fact_subgraphs")
    parser.add_argument("--fact-ids", nargs="*", default=None)
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    src_dsn = os.getenv("DSM_KG_DATABASE_URL")
    if not src_dsn:
        raise SystemExit("DSM_KG_DATABASE_URL not set (load .env)")
    min_db = os.getenv("DSM_MINIMAL_GRAPH_DATABASE_NAME", "dsm5_minimal_graph")
    dsn = target_dsn(src_dsn, min_db)

    fact_ids = args.fact_ids or DEFAULT_FACT_IDS
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(dsn) as conn:
        print(f"Loading graph from {min_db} ...")
        entities, facts, adj, fact_chunks, chunk_facts, chunks = load_graph(conn)
        print(f"  entities={len(entities)} facts={len(facts)} chunks={len(chunks)}")

        # map fact_id -> fact for seed lookup
        fact_by_id = {f["id"]: f for f in facts}

        for fid in fact_ids:
            fact = fact_by_id.get(fid)
            if fact is None:
                print(f"[WARN] fact {fid} not found in DB, skipping")
                continue
            start = [fact["subject_id"], fact["object_id"]]
            visited_entities, subgraph_facts, dist = bfs_subgraph(start, facts, adj, max_hops=args.hops)

            # also collect chunks that mention any visited fact
            mentioned_chunks = set()
            for f in subgraph_facts:
                mentioned_chunks.update(fact_chunks.get(f["id"], []))

            # build JSON-serializable output
            # entity details
            entity_list = [
                {"id": eid, "text": entities[eid]["text"], "type": entities[eid]["type"], "distance": dist.get(eid, None)}
                for eid in visited_entities
            ]
            entity_list.sort(key=lambda x: (x["distance"] if x["distance"] is not None else 99, x["id"]))

            # hop histogram
            hop_hist = defaultdict(int)
            for eid in visited_entities:
                hop_hist[dist.get(eid, -1)] += 1

            out = {
                "seed_fact_id": fid,
                "seed_fact": fact,
                "seed_fact_text": f"{entities[fact['subject_id']]['text']} --{fact['predicate']}--> {entities[fact['object_id']]['text']}",
                "hops": args.hops,
                "stats": {
                    "entities": len(visited_entities),
                    "facts": len(subgraph_facts),
                    "chunks_mentioned": len(mentioned_chunks),
                    "hop_histogram": dict(sorted(hop_hist.items())),
                },
                "entities": entity_list,
                "facts": sorted(subgraph_facts, key=lambda x: x["id"]),
                "fact_chunk_links": {fid2: fact_chunks[fid2] for fid2 in [f["id"] for f in subgraph_facts] if fact_chunks.get(fid2)},
                "chunks": [{"id": cid, "content": chunks[cid]} for cid in sorted(mentioned_chunks)],
            }

            out_path = out_dir / f"{fid}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"  {fid} -> {out_path}  entities={len(visited_entities)} facts={len(subgraph_facts)} chunks={len(mentioned_chunks)} hist={dict(sorted(hop_hist.items()))}")

    print(f"\nDone. {len(fact_ids)} subgraphs in {out_dir}")


if __name__ == "__main__":
    main()
