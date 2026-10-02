"""Eval for DSM sets: constrained-class accuracy vs gold_label.

Replaces eval/eval.py exact-match over KG entities (eval/utils.py:121).
Usage:
    python dsm_eval.py --tog-output ToG_swmh.jsonl --csv ../../dsm_grounded_v1/SWMH/test.csv
    python dsm_eval.py --tog-output ToG_swmh.jsonl --csv ... --labels anxiety "bipolar disorder" depression "no mental disorders" suicide
"""
import argparse
import csv
import json
import re
from collections import Counter


def normalize(s):
    return (s or "").strip().lower().replace("_", " ").strip(" .")


def extract_pred_label(response, labels):
    """Label validated against allowed classes; NULL when unparseable.

    Checks every {..} group (last first), then whole-text substring.
    Never returns free text: garbage like copied entity names -> NULL.
    """
    if not response:
        return "NULL"
    for raw in reversed(re.findall(r"\{([^{}]+)\}", response)):
        cand = normalize(raw)
        for lab in labels:
            if cand == normalize(lab):
                return lab
        for lab in sorted(labels, key=len, reverse=True):
            if normalize(lab) in cand:
                return lab
    low = normalize(response)
    for lab in sorted(labels, key=len, reverse=True):
        if normalize(lab) in low:
            return lab
    return "NULL"


def load_gold(csv_path):
    gold = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold[row["query"]] = row["gold_label"]
            if row.get("benchmark_id"):
                gold[row["benchmark_id"]] = row["gold_label"]
    return gold


def evaluate(tog_output_path, csv_path, labels):
    gold = load_gold(csv_path)
    right = wrong = skipped = 0
    per_label = Counter()
    per_label_ok = Counter()
    conf = Counter()
    with open(tog_output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = d.get("question", "")
            g = gold.get(q, d.get("gold_label", ""))
            if not g:
                skipped += 1
                continue
            pred = extract_pred_label(d.get("results", ""), labels)
            ok = normalize(pred) == normalize(g)
            right += ok
            wrong += (not ok)
            per_label[normalize(g)] += 1
            per_label_ok[normalize(g)] += ok
            conf[(normalize(g), normalize(pred))] += 1
    total = right + wrong
    print(f"total={total} skipped={skipped} acc={right / total:.4f}" if total else "no rows")
    print(f"right={right} wrong={wrong}")
    for lab, n in per_label.most_common():
        print(f"  {lab}: {per_label_ok[lab]}/{n} = {per_label_ok[lab] / n:.3f}")
    return {"acc": right / total if total else 0.0, "right": right,
            "wrong": wrong, "skipped": skipped}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tog-output", required=True,
                    help="ToG_*.jsonl with {question, results, ...}")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--dataset", default="SWMH")
    a = ap.parse_args()
    from dsm_wrapper import DSM_LABELS
    labels = a.labels or DSM_LABELS.get(a.dataset.upper(), DSM_LABELS["SWMH"])
    evaluate(a.tog_output, a.csv, labels)
