"""Full ToG-on-DSM chain for ONE question, saving every step.

Steps: 1 extract entities (LLM) -> 2 Milvus link -> 3 PG beam walk
(depths of relation-prune + entity-score, LLM) -> 4 sufficiency check (LLM)
-> 5 classify into 5 labels (LLM) -> 6 eval vs gold_label.

All prompts, raw LLM replies, scores, chains go to ONE trace JSON.
Secrets from env only: LLM_API_KEY, DSM_KG_DATABASE_URL, EMBEDDING_API_KEY,
MILVUS_URI, MILVUS_TOKEN. Never written to the trace.

  export LLM_API_KEY=... DSM_KG_DATABASE_URL=... EMBEDDING_API_KEY=... MILVUS_URI=... MILVUS_TOKEN=...
  python3 run_trace.py --csv ../../dsm_grounded_v1/SWMH/test.csv --row 0 \
      --out traces/trace_swmh_0.json --width 3 --depth 2
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from config.settings import Settings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsm_wrapper import (SWMH_LABELS, DSM_LABELS, TOPIC_ENTITY_EXTRACT_PROMPT,
                         parse_entity_list, build_dsm_answer_prompt)
from dsm_eval import extract_pred_label, normalize

_SETTINGS = Settings()
LLM_BASE = _SETTINGS.llm_api_base
LLM_MODEL = _SETTINGS.llm_model
RE_BRACE = re.compile(r"\{([^{}]+)\}")


def llm(prompt, trace, tag, max_tokens=256, temperature=0.0, retries=5):
    key = _SETTINGS.llm_api_key
    if not key:
        raise ValueError("LLM_API_KEY is missing from the project Settings/.env")
    body = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    last = ""
    active_ms = 0.0
    for a in range(retries):
        started = time.perf_counter()
        try:
            req = urllib.request.Request(
                LLM_BASE + "/chat/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + key})
            res = json.load(urllib.request.urlopen(req, timeout=180))
            text = res["choices"][0]["message"]["content"]
            active_ms += (time.perf_counter() - started) * 1000
            usage = res.get("usage") or {}
            inp = usage.get("prompt_tokens")
            out = usage.get("completion_tokens")
            _record_llm(trace, {"tag": tag, "prompt": prompt,
                                "response": text, "input_tokens": inp,
                                "output_tokens": out,
                                "latency_ms": round(active_ms, 2)})
            return text
        except Exception as e:
            active_ms += (time.perf_counter() - started) * 1000
            last = str(e)[:200]
            time.sleep(3 * (a + 1))
    _record_llm(trace, {"tag": tag, "prompt": prompt,
                        "response": "", "error": last,
                        "input_tokens": None, "output_tokens": None,
                        "latency_ms": round(active_ms, 2)})
    return ""


def _record_llm(trace, entry):
    trace["llm_calls"].append(entry)
    totals = trace["token_totals"]
    totals["calls"] += 1
    for source, target in (("input_tokens", "input_tokens"),
                           ("output_tokens", "output_tokens")):
        if entry[source] is None:
            totals["usage_complete"] = False
        else:
            totals[target] += entry[source]
    trace["latency_totals"]["llm_ms"] += entry["latency_ms"]


def parse_scored_relations(text):
    out = []
    for m in RE_BRACE.finditer(text or ""):
        inner = m.group(1)
        sm = re.search(r"(.+?)\s*\(Score:\s*([0-9.]+)\)", inner)
        if sm:
            try:
                out.append({"relation": sm.group(1).strip(),
                            "score": float(sm.group(2))})
            except ValueError:
                pass
    return out


def parse_scores(text, n):
    nums = [float(x) for x in re.findall(r"\d+\.\d+|\d+", text or "")]
    if len(nums) == n:
        return nums
    return [1.0 / n] * n if n else []


def parse_sufficient(text):
    """True/False from a sufficiency reply. Brace first, else first word."""
    if not text:
        return False
    m = RE_BRACE.search(text)
    if m and m.group(1).strip().lower() in ("yes", "no"):
        return m.group(1).strip().lower() == "yes"
    for line in text.splitlines():
        s = line.strip().strip('*#"').lower()
        if not s:
            continue
        if s.startswith("yes"):
            return True
        if s.startswith("no"):
            return False
        return False
    return False


def main():
    run_started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--question", default=None,
                    help="ad-hoc question text (skips CSV)")
    ap.add_argument("--gold", default="", help="gold label for ad-hoc runs")
    ap.add_argument("--bench", default="adhoc")
    ap.add_argument("--dataset", default="SWMH",
                    help="SWMH, DR or T-SID (picks the allowed label set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-topics", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=3,
                    help="Milvus hits kept per extracted entity")
    ap.add_argument("--k", type=int, default=10,
                    help="max entities to extract from the post")
    a = ap.parse_args()

    import milvus_linker as ML
    import postgres_func as PG

    if a.question:
        question, gold = a.question, a.gold
        bench = a.bench
    else:
        with open(a.csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        src = rows[a.row]
        question, gold = src["query"], src["gold_label"]
        bench = src.get("benchmark_id", "")

    ALLOWED = DSM_LABELS.get(a.dataset.upper(), SWMH_LABELS)
    trace = {"question": question, "gold_label": gold,
             "benchmark_id": bench, "dataset": a.dataset.upper(),
             "labels": ALLOWED, "llm_model": LLM_MODEL,
             "llm_calls": [], "steps": {}, "depths": [],
             "token_totals": {"input_tokens": 0, "output_tokens": 0,
                              "calls": 0, "usage_complete": True},
             "latency_totals": {"llm_ms": 0.0, "end_to_end_ms": None}}

    task_ctx = ("Note: the final task is classification into ONE of these "
                f"classes: {'; '.join(ALLOWED)}.")
    # 1. extract
    p1 = TOPIC_ENTITY_EXTRACT_PROMPT.format(
        k=a.k, post=question[:3000], question=question[-500:],
        labels="; ".join(ALLOWED))
    names = parse_entity_list(llm(p1, trace, "1_extract"))
    trace["steps"]["extracted_names"] = names

    # 2. link (top-k most similar per extracted entity, round-robin
    # so the walk starts from every name, not just the first one)
    qnames = names[:a.max_topics] or names
    hit_lists = ML.ranked(qnames, top_k=a.top_k) if qnames else []
    def _take(hits, topic, seen_text, allow_dup_text):
        for h in hits:
            if h["id"] in topic:
                continue
            key = (h["text"] or "").strip().lower()
            if key in seen_text and not allow_dup_text:
                continue  # same text, different id: identical neighborhood
            seen_text.add(key)
            topic[h["id"]] = h["text"]
            return True
        return False

    topic, seen_topic_text = {}, set()
    # pass 1: guarantee >=1 node per extracted entity
    for hits in hit_lists:
        if not _take(hits, topic, seen_topic_text, False):
            _take(hits, topic, seen_topic_text, True)
    # pass 2: round-robin the rest up to top-k per entity
    for i in range(a.top_k):
        for hits in hit_lists:
            if i < len(hits):
                _take(hits[i:i + 1], topic, seen_topic_text, False)
    trace["steps"]["topic_entity"] = topic  # {pg_id: text}
    trace["steps"]["link_top_k"] = a.top_k
    trace["steps"]["link_hits"] = {
        n: [(h["id"], h["text"], round(h["distance"], 4)) for h in hits]
        for n, hits in zip(qnames, hit_lists)}
    if not topic:
        trace["final"] = {"pred_label": "NULL", "correct": False,
                          "note": "no linked entities"}
        _save(a.out, trace, run_started)
        return

    # 3. beam walk (loop-free: never revisit an entity id)
    chains_all, cur = [], topic
    visited = set(topic)  # pg ids already in the paths
    for d in range(1, a.depth + 1):
        dinfo = {"depth": d, "expansions": [], "kept": [],
                 "skipped_visited": 0}
        scored = []
        for eid, ename in list(cur.items())[:a.max_topics]:
            head_rels, tail_rels = PG.get_relations(eid)
            cands = sorted(set(head_rels + tail_rels))[:60]
            if not cands:
                continue
            pr = (f"Retrieve {a.width} relations (separated by semicolon) that help answer "
                  f"the question, score 0-1 (sum=1).\n{task_ctx}\nQ: {question}\nTopic Entity: {ename}\n"
                  f"Relations: {'; '.join(cands)}\nA: use ONLY relations copied exactly "
                  f"from the list above, about the Topic Entity, as {{rel (Score: s)}} + one-line reason:")
            reply = llm(pr, trace, f"3_d{d}_rel_{ename}")
            picked = [p for p in parse_scored_relations(reply)
                      if p["relation"] in cands][:a.width]
            if not picked:
                dinfo["expansions"].append({"entity": ename, "picked": [],
                                            "note": "prune-invalid",
                                            "raw": (reply or "")[:300]})
                continue
            for p_ in picked:
                p_["dir"] = "out" if p_["relation"] in head_rels else "in"
            for pr_ in picked:
                rel = pr_["relation"]
                is_head = rel in head_rels
                nbrs = PG.get_neighbors(eid, rel, head=is_head, limit=12)
                if not nbrs:
                    continue
                cand_names = [f"{n} (via {v['entry']})" if v else n
                              for _, n, v in nbrs]
                ps = (f"Score entities 0-1 (sum=1) for the question.\n{task_ctx}\nQ: {question}\n"
                      f"Relation: {rel}\nEntities: {'; '.join(cand_names)}\nScore (comma list):")
                scores = parse_scores(llm(ps, trace, f"3_d{d}_ent_{rel}"), len(cand_names))
                for (nid, nname, via), s in zip(nbrs, scores):
                    if nid in visited or nid == eid:
                        dinfo["skipped_visited"] += 1
                        continue
                    # directed edge: subject -rel-> object (topic may be either end)
                    if is_head:
                        subj, obj = ename, nname
                    else:
                        subj, obj = nname, ename
                    scored.append({"score": s * pr_["score"],
                                   "subj": subj, "relation": rel, "obj": obj,
                                   "next_id": nid, "next": nname,
                                   "via": via,
                                   "dir": "out" if is_head else "in"})
                dinfo["expansions"].append({"entity": ename, "picked": picked,
                                            "neighbors": cand_names, "scores": scores})
        scored.sort(key=lambda x: -x["score"])
        kept, seen_path = [], set()
        for s in scored:
            sig = (s["subj"], s["relation"], s["obj"], s["dir"],
                   tuple(s["via"]["trips"]) if s["via"] else ())
            if (s["score"] <= 0 or sig in seen_path
                    or s["next_id"] in visited):
                continue
            seen_path.add(sig)
            kept.append(s)
            if len(kept) >= a.width:
                break
        dinfo["kept"] = kept
        trace["depths"].append(dinfo)
        if not kept:
            break
        depth_trips = []
        for s in kept:
            visited.add(s["next_id"])
            if s["via"]:
                v = s["via"]
                # splice intermediate sub-path inline: full path, same depth
                if s["dir"] == "out":
                    depth_trips.append(
                        (s["subj"], s["relation"], v["entry"], "out"))
                else:
                    depth_trips.append(
                        (v["entry"], s["relation"], s["obj"], "in"))
                depth_trips.extend(v["trips"])
                visited.update(v["pass_ids"])
                dinfo["intermediate_resolved"] = \
                    dinfo.get("intermediate_resolved", 0) + 1
            else:
                depth_trips.append(
                    (s["subj"], s["relation"], s["obj"], s["dir"]))
        chains_all.append(depth_trips)
        # 4. sufficiency
        trip = "\n".join(", ".join(map(str, ch[:3]))
                         for sub in chains_all for ch in sub)
        pc = (f"Given the post and triplets, is it sufficient to classify into one of "
              f"{'; '.join(ALLOWED)}? First line of your answer must be "
              f"exactly {{Yes}} or {{No}}. Then give the reason.\n"
              f"Post: {question}\nTriplets: {trip}\nA:")
        chk = llm(pc, trace, f"4_d{d}_sufficient")
        trace["steps"].setdefault("sufficient_votes", {})[str(d)] = chk[:120]
        if parse_sufficient(chk):
            trace["steps"]["stopped_at_depth"] = d
            break
        cur = {s["next_id"]: s["next"] for s in kept}

    # 5. classify
    trip = "\n".join(", ".join(map(str, ch[:3]))
                     for sub in chains_all for ch in sub) or "(no triplets found)"
    pa = build_dsm_answer_prompt(question, trip, ALLOWED)
    final_raw = llm(pa, trace, "5_classify", max_tokens=300)
    pred = extract_pred_label(final_raw, ALLOWED)

    # 6. eval
    trace["steps"]["chains"] = chains_all
    trace["final"] = {"raw_answer": final_raw, "pred_label": pred,
                      "gold_label": gold,
                      "correct": normalize(pred) == normalize(gold)}
    _save(a.out, trace, run_started)
    print(f"pred={pred} gold={gold} correct={trace['final']['correct']} -> {a.out}")


def _save(path, trace, run_started):
    trace["latency_totals"]["llm_ms"] = round(trace["latency_totals"]["llm_ms"], 2)
    trace["latency_totals"]["end_to_end_ms"] = round((time.perf_counter() - run_started) * 1000, 2)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
