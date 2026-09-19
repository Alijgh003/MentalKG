#!/usr/bin/env python3
"""Extract explicit class labels from MentalLLaMA golden responses.

The upstream instruction files store the target label and its explanation in
the ``gpt-3.5-turbo`` column.  This script preserves every source column and
adds a normalized ``label`` column to derived CSV files.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Final


DEFAULT_INPUT_DIR: Final = Path(
    "datasets/external/MentalLLaMA/test_data/test_instruction"
)
DEFAULT_OUTPUT_DIR: Final = Path(
    "datasets/processed/MentalLLaMA/test_instruction_labeled"
)

DATASET_FILES: Final = {
    "dreaddit": "dreaddit.csv",
    "t-sid": "t-sid.csv",
    "SAD": "SAD.csv",
    "MultiWD": "MultiWD.csv",
}

LABEL_PREFIXES: Final = {
    "dreaddit": {
        "Yes, the poster suffers from stress.": "yes",
        "No, the poster does not suffer from stress.": "no",
    },
    "t-sid": {
        "This post shows depression.": "depression",
        "This post shows suicide or self-harm tendency.": "suicide_or_self_harm",
        "This post shows PTSD.": "ptsd",
        "This post shows no mental disorders.": "no_mental_disorder",
    },
    "SAD": {
        "This post shows the stress cause related to school.": "school",
        "This post shows the stress cause related to financial problem.": "financial_problem",
        "This post shows the stress cause related to family issues.": "family_issues",
        "This post shows the stress cause related to social relationships.": "social_relationships",
        "This post shows the stress cause related to work.": "work",
        "This post shows the stress cause related to health issues.": "health_issues",
        "This post shows the stress cause related to emotion turmoil.": "emotion_turmoil",
        "This post shows the stress cause related to everyday decision making.": "everyday_decision_making",
        "This post shows other stress causes.": "other_causes",
    },
    "MultiWD": {
        "Yes, this wellness dimension exists in the post.": "yes",
        "No, this wellness dimension does not exist in the post.": "no",
    },
}


def extract_label(dataset: str, golden_response: str) -> str:
    """Return the normalized label encoded before ``Reasoning:``."""
    prefix = golden_response.split("Reasoning:", maxsplit=1)[0].strip()
    try:
        return LABEL_PREFIXES[dataset][prefix]
    except KeyError as exc:
        raise ValueError(
            f"Unrecognized golden-response label for {dataset}: {prefix!r}"
        ) from exc


def process_dataset(dataset: str, input_path: Path, output_path: Path) -> Counter[str]:
    """Add a label column to one MentalLLaMA CSV and return label counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "gpt-3.5-turbo" not in reader.fieldnames:
            raise ValueError(f"Missing 'gpt-3.5-turbo' column in {input_path}")

        fieldnames = [name for name in reader.fieldnames if name != "label"] + ["label"]
        counts: Counter[str] = Counter()

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            for row_number, row in enumerate(reader, start=2):
                try:
                    label = extract_label(dataset, row["gpt-3.5-turbo"])
                except ValueError as exc:
                    raise ValueError(f"{input_path}:{row_number}: {exc}") from exc
                row["label"] = label
                writer.writerow(row)
                counts[label] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset, filename in DATASET_FILES.items():
        input_path = args.input_dir / filename
        output_path = args.output_dir / filename
        counts = process_dataset(dataset, input_path, output_path)
        distribution = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
        print(f"{dataset}: rows={sum(counts.values())}; labels: {distribution}")


if __name__ == "__main__":
    main()
