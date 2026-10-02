#!/usr/bin/env python3
"""Join directed seed paths when an object endpoint feeds a subject endpoint."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def directed_paths(paths):
    """Keep only paths traversed entirely subject->object."""
    return [
        path for path in paths
        if path.get("hops", 0) > 0
        and all(direction == "subject_to_object" for direction in path.get("traversal_directions", []))
    ]


def join_paths(left, right):
    """Concatenate two paths sharing left.object == right.subject."""
    if left["nodes"][-1] != right["nodes"][0]:
        return None
    nodes = left["nodes"] + right["nodes"][1:]
    if len(nodes) != len(set(nodes)):
        return None
    return {
        "start_entity_id": nodes[0],
        "end_entity_id": nodes[-1],
        "nodes": nodes,
        "fact_ids": left["fact_ids"] + right["fact_ids"],
        "predicates": left["predicates"] + right["predicates"],
        "traversal_directions": left["traversal_directions"] + right["traversal_directions"],
        "node_texts": left.get("node_texts", []) + right.get("node_texts", [])[1:],
        "node_types": left.get("node_types", []) + right.get("node_types", [])[1:],
        "fact_texts": left.get("fact_texts", []) + right.get("fact_texts", []),
        "hops": left["hops"] + right["hops"],
        "joined_from": [left.get("path_id"), right.get("path_id")],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-segments", type=int, default=2)
    parser.add_argument("--segment-hops", type=int, default=None,
                        help="use only paths with exactly this many hops")
    parser.add_argument("--max-output-paths", type=int, default=10000)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    candidates = directed_paths(data.get("paths", []))
    if args.segment_hops is not None:
        candidates = [path for path in candidates if path.get("hops") == args.segment_hops]
    for index, path in enumerate(candidates):
        path["path_id"] = index
    by_start = defaultdict(list)
    for path in candidates:
        by_start[path["nodes"][0]].append(path)

    joined = []
    for left in candidates:
        for right in by_start.get(left["nodes"][-1], []):
            chain = join_paths(left, right)
            if chain is not None:
                joined.append(chain)

    # Deduplicate exact chains and keep the shortest/first representation.
    unique = {}
    for chain in joined:
        unique.setdefault(tuple(chain["fact_ids"]), chain)
    joined = list(unique.values())
    joined.sort(key=lambda path: (path["hops"], path["nodes"]))
    total_joined = len(joined)
    joined = joined[: args.max_output_paths]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": args.input,
        "rule": "left object == right subject; all traversals subject_to_object; no repeated nodes",
        "summary": {
            "input_paths": len(data.get("paths", [])),
            "directed_paths": len(candidates),
            "joined_two_segment_paths": total_joined,
            "written_paths": len(joined),
        },
        "paths": joined,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
