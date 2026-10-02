"""DSPy/trace runner for the DSM adaptation of ToG-2 (original clone untouched).

One output JSON per benchmark row, with the same top-level structure as
ToG-1's run_dspy.py trace. Every LLM decision uses a typed DSPy Signature.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

from dsm_dataset import DSM_LABELS, load_frozen_csv, normalize_dataset_name
from dsm_dspy_programs import DSMToG2Program, configure_dspy_lm
from dsm_dspy_trace import new_trace, traced_call
from dsm_entity_context import rank_candidate_entities
from dsm_seed_context import _tog1_backends, rank_seed_chunks


def _label(value: object, allowed: Sequence[str]) -> str | None:
    if not isinstance(value, str):
        return None
    return next((item for item in allowed if value.strip().casefold() == item.casefold()), None)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def prepare_one_record(record: dict, program: DSMToG2Program, trace: dict,
                       max_seed_nodes: int = 10) -> dict:
    """DSPy extraction + ToG-1 round-robin Jina/Milvus entity linking."""
    prediction = traced_call(
        trace, "1_extract", program.extractor,
        post=record["raw_question"], allowed_labels=record["allowed_labels"],
        k=max_seed_nodes,
    )
    values = prediction.entities
    if not isinstance(values, list):
        raise ValueError("DSPy extraction must return entities as a list")
    seen: set[str] = set()
    names: list[str] = []
    extraction_pruned: list[dict] = []
    for value in values:
        name = str(value).strip().strip("{}").strip()
        if not name:
            extraction_pruned.append({"value": str(value), "reason": "empty"})
        elif name.casefold() in seen:
            extraction_pruned.append({"value": str(value), "reason": "duplicate"})
        elif len(names) >= max_seed_nodes:
            extraction_pruned.append({"value": str(value), "reason": "seed_cap"})
        else:
            names.append(name)
            seen.add(name.casefold())
    trace["steps"]["extracted_names"] = names
    trace["steps"]["extraction_pruned"] = extraction_pruned

    _, ml = _tog1_backends()
    hit_lists = ml.ranked(names, top_k=3) if names else []
    topic: dict[str, str] = {}
    selected_text: set[str] = set()
    choices: list[dict] = []

    def take(hits: Sequence[dict], surface: str, allow_duplicate_text: bool,
             pass_name: str) -> bool:
        for hit in hits:
            if len(topic) >= max_seed_nodes:
                return False
            entity_id = str(hit["id"])
            entity_name = str(hit["text"])
            key = entity_name.strip().casefold()
            if entity_id in topic or (key in selected_text and not allow_duplicate_text):
                continue
            topic[entity_id] = entity_name
            selected_text.add(key)
            choices.append({"surface": surface, "id": entity_id,
                            "name": entity_name, "similarity": hit.get("distance"),
                            "pass": pass_name})
            return True
        return False

    for name, hits in zip(names, hit_lists):
        take(hits, name, False, "one_per_surface") or take(hits, name, True, "one_per_surface")
    for rank in range(3):
        for name, hits in zip(names, hit_lists):
            if rank < len(hits):
                take(hits[rank:rank + 1], name, False, "round_robin")

    trace["steps"]["link_top_k"] = 3
    trace["steps"]["link_hits"] = {
        name: [{"id": str(hit["id"]), "name": hit["text"],
                "similarity": hit.get("distance")}
               for hit in hits]
        for name, hits in zip(names, hit_lists)
    }
    trace["steps"]["link_selected"] = choices
    trace["steps"]["link_pruned"] = [
        {"surface": name, "id": str(hit["id"]), "name": hit["text"],
         "similarity": hit.get("distance"),
         "reason": ("duplicate_entity" if str(hit["id"]) in topic else
                    "seed_cap" if len(topic) >= max_seed_nodes else
                    "duplicate_name" if str(hit["text"]).strip().casefold() in selected_text else
                    "lower_rank")}
        for name, hits in zip(names, hit_lists) for hit in hits
        if str(hit["id"]) not in topic
    ]
    trace["steps"]["topic_entity"] = topic
    record = dict(record)
    record["extracted_entities"] = names
    record["qid_topic_entity"] = topic
    record["entity_preparation_status"] = "linked" if topic else "no_link"
    return record


def _topic_prune(question: str, seeds: dict[str, str], labels: list[str],
                 width: int, program: DSMToG2Program, trace: dict) -> dict[str, str]:
    if len(seeds) <= width:
        selected = dict(seeds)
        trace["steps"]["topic_prune"] = {"input": seeds, "selected": selected,
                                            "pruned": [], "llm_bypassed": True}
        return selected
    prediction = traced_call(
        trace, "2_topic_prune", program.topic_pruner,
        question=question, allowed_labels=labels, seed_nodes=_json(seeds), width=width,
    )
    if not isinstance(prediction.selected_ids, list):
        raise ValueError("DSPy topic prune must return selected_ids as a list")
    chosen_ids: list[str] = []
    invalid: list[str] = []
    for value in prediction.selected_ids:
        entity_id = str(value)
        if entity_id not in seeds or entity_id in chosen_ids:
            invalid.append(entity_id)
        elif len(chosen_ids) < width:
            chosen_ids.append(entity_id)
        else:
            invalid.append(entity_id)
    selected = {entity_id: seeds[entity_id] for entity_id in chosen_ids}
    trace["steps"]["topic_prune"] = {
        "input": seeds, "selected": selected,
        "pruned": [{"id": entity_id, "name": name, "reason": "llm_not_selected"}
                   for entity_id, name in seeds.items() if entity_id not in selected],
        "invalid_output_ids": invalid, "llm_bypassed": False,
    }
    return selected


def _relation_options(frontier: Sequence[dict], pg) -> tuple[list[dict], list[dict]]:
    options: list[dict] = []
    backtracks: list[dict] = []
    for node in frontier:
        outgoing, incoming = pg.get_relations(node["id"])
        blocked_head = not node.get("head", False)
        for head, relations in ((True, outgoing), (False, incoming)):
            for relation in sorted(set(relations)):
                if relation == node.get("relation") and head == blocked_head:
                    backtracks.append({"entity_id": str(node["id"]),
                                       "entity_name": node["name"],
                                       "relation": relation, "head": head,
                                       "reason": "immediate_backtrack"})
                    continue
                options.append({"option_id": f"r{len(options)}",
                                "entity_id": str(node["id"]),
                                "entity_name": node["name"],
                                "relation": relation, "head": head})
    return options, backtracks


def _prune_relations(question: str, labels: list[str], frontier: Sequence[dict],
                     width: int, pg, program: DSMToG2Program, trace: dict,
                     depth: int) -> tuple[list[dict], dict]:
    options, backtracks = _relation_options(frontier, pg)
    if not options:
        return [], {"options": [], "selected": [], "pruned": backtracks,
                    "invalid_output_ids": [], "llm_bypassed": True}
    prediction = traced_call(
        trace, f"3_d{depth}_relations", program.relation_pruner,
        question=question, allowed_labels=labels,
        relation_options=_json(options), width=width,
    )
    ids, scores = prediction.selected_option_ids, prediction.scores
    if not isinstance(ids, list) or not isinstance(scores, list) or len(ids) != len(scores):
        raise ValueError("DSPy relation prune IDs and scores must be aligned lists")
    by_option = {item["option_id"]: item for item in options}
    proposed: dict[str, float] = {}
    invalid: list[dict] = []
    for option_id, score in zip(ids, scores):
        option_id = str(option_id)
        try:
            number = float(score)
        except (TypeError, ValueError):
            invalid.append({"option_id": option_id, "score": score, "reason": "invalid_score"})
            continue
        if option_id not in by_option or not 0 <= number <= 1:
            invalid.append({"option_id": option_id, "score": number,
                            "reason": "unknown_option_or_out_of_range"})
            continue
        proposed[option_id] = max(number, proposed.get(option_id, 0.0))
    by_entity: dict[str, list[dict]] = {}
    for option_id, score in proposed.items():
        if score >= 0.2:
            option = {**by_option[option_id], "score": score}
            by_entity.setdefault(option["entity_id"], []).append(option)
    selected = [item for group in by_entity.values()
                for item in sorted(group, key=lambda item: -item["score"])[:width]]
    chosen_ids = {item["option_id"] for item in selected}
    pruned = backtracks + [
        {**option, "score": proposed.get(option["option_id"]),
         "reason": ("llm_not_selected" if option["option_id"] not in proposed else
                    "below_0.2_threshold" if proposed[option["option_id"]] < 0.2 else
                    "over_width")}
        for option in options if option["option_id"] not in chosen_ids
    ]
    return selected, {"options": options, "selected": selected,
                      "pruned": pruned, "invalid_output_ids": invalid,
                      "llm_bypassed": False}


def _discover_candidates(relations: Sequence[dict], pg, visited: set[str],
                         frontier: Sequence[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    parents = {str(node["id"]): node for node in frontier}
    candidates: list[dict] = []
    skipped: list[dict] = []
    expansions: list[dict] = []
    for relation in relations:
        parent = parents[relation["entity_id"]]
        neighbors = pg.get_neighbors(
            relation["entity_id"], relation["relation"],
            head=relation["head"], limit=50, resolve_pass=True,
        )
        expansion = {"relation": relation, "neighbor_limit": 50,
                     "neighbors": [], "candidate_ids": []}
        for entity_id, name, via in neighbors:
            entity_id = str(entity_id)
            expansion["neighbors"].append({"id": entity_id, "name": name, "via": via})
            reason = None
            if entity_id in visited:
                reason = "already_visited"
            elif not name:
                reason = "missing_entity_name"
            elif via and any(str(intermediate) in visited for intermediate in via["pass_ids"]):
                reason = "intermediate_already_visited"
            if reason:
                skipped.append({"id": entity_id, "name": name,
                                "relation_option_id": relation["option_id"], "reason": reason})
                continue
            if relation["head"]:
                first = f"{relation['entity_name']} {relation['relation']} {via['entry'] if via else name}."
            else:
                first = f"{via['entry'] if via else name} {relation['relation']} {relation['entity_name']}."
            additional = [f"{subject} {predicate} {obj}."
                          for subject, predicate, obj, _ in via["trips"]] if via else []
            path = [*parent.get("path", []), first, *additional]
            candidates.append({
                "id": entity_id, "name": name,
                "topic_entities": relation["entity_name"],
                "topic_id": relation["entity_id"],
                "relation": relation["relation"],
                "head": relation["head"], "path": path,
                "triple_sentence": " ".join(path), "via": via,
            })
            expansion["candidate_ids"].append(entity_id)
        expansions.append(expansion)
    return candidates, expansions, skipped


def _initial_context(chunks: Sequence[dict], topic: dict[str, str]) -> list[dict]:
    return [
        {"chunk_id": item["chunk_id"], "text": item["text"],
         "entity_names": [topic[entity_id] for entity_id in item["selected_seed_ids"]],
         "path": [], "score": item["score"]}
        for item in chunks
    ]


def _candidate_context(top_context: Sequence[dict], candidates: Sequence[dict]) -> list[dict]:
    return [
        {"chunk_id": item["chunk_id"], "text": item["chunk"],
         "entity_names": [item["entity_name"]], "entity_id": item["entity_id"],
         "path": candidates[item["candidate_index"]]["path"], "score": item["score"]}
        for item in top_context
    ]


def _knowledge_inputs(selected: Sequence[dict], contexts: Sequence[dict]) -> dict[str, str]:
    return {
        "selected_triple_paths": _json([
            {"entity_id": item["id"], "entity_name": item["name"], "path": item["path"]}
            for item in selected
        ]),
        "ranked_contexts": _json([
            {"rank": index, "entity_names": item["entity_names"],
             "entity_id": item.get("entity_id"), "path": item["path"],
             "chunk_id": item["chunk_id"], "chunk": item["text"]}
            for index, item in enumerate(contexts, start=1)
        ]),
    }


def _sufficiency(program: DSMToG2Program, trace: dict, tag: str,
                 question: str, labels: list[str], selected: Sequence[dict],
                 contexts: Sequence[dict], clue: str) -> dict:
    prediction = traced_call(
        trace, tag, program.checker,
        question=question, allowed_labels=labels, previous_clue=clue,
        **_knowledge_inputs(selected, contexts),
    )
    if not isinstance(prediction.sufficient, bool):
        raise ValueError("DSPy sufficiency output must be a boolean")
    label = _label(prediction.label, labels)
    if prediction.sufficient and not label:
        raise ValueError("DSPy said sufficient without an allowed class label")
    sufficiency_explanation = str(prediction.sufficiency_explanation or "")
    answer_explanation = str(prediction.answer_explanation or "")
    return {"sufficient": prediction.sufficient,
            "label": label if prediction.sufficient else None,
            "clue": str(prediction.clue or ""),
            "sufficiency_explanation": sufficiency_explanation,
            "answer_explanation": answer_explanation,
            # Backward-compatible field consumed by existing trace adapters.
            "explanation": answer_explanation if prediction.sufficient else sufficiency_explanation}


def _rewrite(program: DSMToG2Program, trace: dict, tag: str,
             question: str, labels: list[str], clue: str,
             selected: Sequence[dict], contexts: Sequence[dict]) -> str:
    prediction = traced_call(
        trace, tag, program.rewriter,
        question=question, allowed_labels=labels, clue=clue,
        **_knowledge_inputs(selected, contexts),
    )
    query = str(prediction.query or "").strip()
    if not query:
        raise ValueError("DSPy query reformulation returned an empty query")
    trace["steps"].setdefault("query_rewrites", []).append({
        "tag": tag, "clue": clue,
        "needed_evidence": str(prediction.needed_evidence or ""),
        "query": query,
    })
    return query + "\nAllowed classes: " + "; ".join(labels)


def _finish(trace: dict, label: str | None, explanation: str,
            end_mode: str, stop_reason: str, stopped_at_depth: int | None = None) -> dict:
    gold = trace.get("gold_label", "")
    pred = label or "NULL"
    trace["steps"]["chains"] = [
        [item["path"] for item in depth.get("kept", [])]
        for depth in trace["depths"]
    ]
    if stopped_at_depth is not None:
        trace["steps"]["stopped_at_depth"] = stopped_at_depth
    trace["final"] = {
        "raw_answer": f"{explanation}\n\n{{{pred}}}" if label else explanation,
        "pred_label": pred, "gold_label": gold,
        "correct": bool(gold) and pred.casefold() == str(gold).casefold(),
        "end_mode": end_mode, "stop_reason": stop_reason,
    }
    return trace


def run_question(record: dict, *, program: DSMToG2Program, trace: dict,
                 width: int = 5, depth: int = 5, max_seed_nodes: int = 10) -> dict:
    """Run the ToG-2 loop and keep exact before/after decisions in a ToG-1 trace."""
    if width < 1 or depth < 0 or max_seed_nodes < width:
        raise ValueError("Require width >= 1, depth >= 0 and max_seed_nodes >= width")
    question, labels = record["question"], record["allowed_labels"]
    seeds = dict(list(record.get("qid_topic_entity", {}).items())[:max_seed_nodes])
    if not seeds and record.get("entity_preparation_status") == "not_run":
        raise ValueError("Run DSPy extraction/linking before the ToG-2 loop")
    topic = _topic_prune(question, seeds, labels, width, program, trace)
    selected_chunks, all_chunks = (
        rank_seed_chunks(question, topic, top_k_per_seed=3, return_all=True)
        if topic else ([], [])
    )
    initial_context = _initial_context(selected_chunks, topic)
    kept_chunk_ids = {item["chunk_id"] for item in selected_chunks}
    trace["steps"]["initial_context"] = {
        "all_scored_chunks": all_chunks, "selected": initial_context,
        "pruned": [{"chunk_id": item["chunk_id"], "score": item["score"],
                    "reason": "outside_top_3_for_each_seed"}
                   for item in all_chunks if item["chunk_id"] not in kept_chunk_ids],
    }
    initial_vote = _sufficiency(program, trace, "4_d0_sufficient", question,
                                 labels, [], initial_context, "")
    trace["steps"].setdefault("sufficient_votes", {})["0"] = initial_vote
    if initial_vote["sufficient"]:
        return _finish(trace, initial_vote["label"], initial_vote["explanation"],
                       "initial_context_sufficient", "llm_sufficient", 0)

    clue = initial_vote["clue"]
    query = _rewrite(program, trace, "4_d0_query", question, labels,
                     clue, [], initial_context)
    pg, _ = _tog1_backends()
    frontier = [{"id": entity_id, "name": name, "path": []}
                for entity_id, name in topic.items()]
    visited = set(topic)
    selected: list[dict] = []
    contexts = initial_context
    stop_reason = "max_depth"

    for d in range(1, depth + 1):
        if not frontier:
            stop_reason = "no_topic_entities"
            break
        relations, rp = _prune_relations(query, labels, frontier, width,
                                        pg, program, trace, d)
        candidates, expansions, skipped = _discover_candidates(relations, pg,
                                                                 visited, frontier)
        dinfo = {"depth": d, "query": query, "frontier": frontier,
                 "relation_options": rp["options"],
                 "selected_relations": relations,
                 "pruned_relations": rp["pruned"],
                 "invalid_relation_output": rp["invalid_output_ids"],
                 "expansions": expansions, "candidates": candidates,
                 "pruned_candidates": skipped,
                 "skipped_visited": sum(item["reason"] == "already_visited"
                                        for item in skipped),
                 "kept": []}
        trace["depths"].append(dinfo)
        if not candidates:
            stop_reason = "no_candidates"
            break
        selected, top_context, details = rank_candidate_entities(
            query, candidates, width=width, context_number=10,
            alpha=0.8, return_all=True,
        )
        contexts = _candidate_context(top_context, candidates)
        dinfo["all_scored_contexts"] = details["all_scored_contexts"]
        dinfo["top_context"] = contexts
        dinfo["pruned_contexts"] = [
            {**item, "reason": "outside_global_top_10"}
            for item in details["all_scored_contexts"][10:]
        ]
        dinfo["entity_ranking"] = details["entity_ranking"]
        dinfo["kept"] = selected
        dinfo["pruned_entities"] = [
            item for item in details["entity_ranking"] if not item["selected"]
        ]
        if not selected:
            stop_reason = "no_contexts"
            break
        vote = _sufficiency(program, trace, f"4_d{d}_sufficient", question,
                            labels, selected, contexts, clue)
        dinfo["reasoning"] = vote
        trace["steps"]["sufficient_votes"][str(d)] = vote
        if vote["sufficient"]:
            return _finish(trace, vote["label"], vote["explanation"],
                           "reasoning_sufficient", "llm_sufficient", d)
        clue = vote["clue"]
        visited.update(item["id"] for item in selected)
        for item in selected:
            if item.get("via"):
                visited.update(str(node_id) for node_id in item["via"]["pass_ids"])
        frontier = selected
        if d < depth:
            query = _rewrite(program, trace, f"4_d{d}_query", question,
                             labels, clue, selected, contexts)

    best = traced_call(
        trace, "5_best_effort", program.final_answerer,
        question=question, allowed_labels=labels, previous_clue=clue,
        **_knowledge_inputs(selected, contexts),
    )
    label = _label(best.label, labels)
    return _finish(trace, label, str(best.explanation or ""),
                   "best_effort" if label else "failure", stop_reason)


def main() -> int:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("SWMH", "T-SID", "TSID", "DR"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="frozen DSM CSV; extract/link with DSPy and log all calls")
    source.add_argument("--prepared", help="prepared JSON; historical extraction tokens unavailable")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()
    if args.row < 0:
        parser.error("--row must be non-negative")
    dataset = normalize_dataset_name(args.dataset)
    if args.csv:
        records = load_frozen_csv(args.csv, dataset, limit=args.row + 1)
    else:
        with open(args.prepared, encoding="utf-8") as handle:
            records = json.load(handle)
    if args.row >= len(records):
        raise IndexError(f"row {args.row} outside {len(records)} records")
    record = records[args.row]
    if record.get("dataset") != dataset or record.get("allowed_labels") != DSM_LABELS[dataset]:
        raise ValueError("Input record does not match the requested DSM dataset and labels")

    lm = configure_dspy_lm()
    program = DSMToG2Program()
    trace = new_trace(record, lm.model)
    if args.prepared:
        trace["steps"]["entity_preparation_source"] = "prepared_file"
        trace["steps"]["historical_extraction_token_usage"] = "unavailable"
        trace["token_totals"]["preparation_covered"] = False
        trace["steps"]["extracted_names"] = record.get("extracted_entities", [])
        trace["steps"]["topic_entity"] = record.get("qid_topic_entity", {})
    else:
        trace["token_totals"]["preparation_covered"] = True
    output = Path(args.output)
    try:
        if args.csv:
            record = prepare_one_record(record, program, trace)
            trace["steps"]["entity_preparation_source"] = "dspy_live"
        run_question(record, program=program, trace=trace,
                     width=args.width, depth=args.depth)
    except Exception as exc:
        message = str(exc).lower()
        model_format_error = any(marker in message for marker in (
            "ids and scores must be aligned lists",
            "jsonadapter failed to parse",
            "failed to parse the lm response",
            "lm response cannot be serialized to a json object",
            "expected to find output fields in the lm response",
            "lm returned an empty or null response",
        ))
        trace["final"] = {"pred_label": "NULL", "gold_label": record.get("gold_label", ""),
                          "correct": False, "end_mode": "error", "error": str(exc),
                          "error_category": "model_output_format" if model_format_error else "runtime_error"}
        _save_trace(output, trace, run_started)
        raise
    _save_trace(output, trace, run_started)
    print(f"pred={trace['final']['pred_label']} gold={trace['gold_label']} "
          f"correct={trace['final']['correct']} -> {output}")
    return 0


def _save_trace(output: Path, trace: dict, run_started: float) -> None:
    trace["latency_totals"]["end_to_end_ms"] = round(
        (time.perf_counter() - run_started) * 1000, 2
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(trace, handle, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
