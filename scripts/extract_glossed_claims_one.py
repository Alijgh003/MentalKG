#!/usr/bin/env python3
"""Extract gloss-enriched claims for the requested resentment/envy explanation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.local_graph_eval import LocalGraphEvaluator


EXPLANATION = """The post expresses feelings of resentment, envy, and inadequacy, which can be indicative of depression. The poster compares their own situation unfavorably to their friend's, highlighting a lack of job offers, lower income, inability to afford travel, and difficulty in relationships. These experiences contribute to the poster's feelings of inadequacy and their desire to distance themselves from their friend for the sake of their mental well-being. While the emotions expressed in the post align with some depressive symptoms, the intensity and overall tone of the post do not suggest very severe depression. The focus on the impact of the friend's achievements and comparison to their own life is more specific to feelings of jealousy and self-doubt, rather than overwhelming emotional distress often associated with severe depression."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/evaluations/glossed_claim_extraction_resentment_envy.json",
    )
    parser.add_argument(
        "--explanation",
        default=EXPLANATION,
        help="Explanation text to extract; defaults to the original resentment/envy example.",
    )
    args = parser.parse_args()
    evaluator = LocalGraphEvaluator()
    entities, claims, usage = evaluator.extract(args.explanation, "GLOSS")
    payload = {
        "source": "user_provided_explanation",
        "explanation": args.explanation,
        "entities": entities,
        "claims": [claim.as_dict() for claim in claims],
        "usage": usage,
        "prompt_version": "gloss-general-entities-v1",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "claims": len(claims), "usage": usage}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
