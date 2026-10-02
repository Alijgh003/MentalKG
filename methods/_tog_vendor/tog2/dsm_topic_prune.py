"""Topic Prune adapter for DSM seed nodes in ToG-2.

The DSM configuration takes up to 10 linked seed nodes and retains at most
``width=5`` suitable starting nodes. IDs are PostgreSQL entity UUIDs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping

from prompt_list import topic_prune_demos


def build_topic_prune_prompt(
    question: str, seed_nodes: Mapping[str, str], width: int = 5
) -> str:
    """Use ToG-2's topic-prune examples with DSM graph terminology."""
    demos = topic_prune_demos.replace(
        "Wikipedia knowledge graph", "DSM knowledge graph"
    ).replace("wiki knowledge graph", "DSM knowledge graph")
    return (
        demos
        + "\nThe IDs below are PostgreSQL entity UUIDs. Select at most "
        + str(width)
        + " starting entities from the provided IDs and copy each selected "
        "name exactly. The question includes all allowed classification "
        "labels. Return only one JSON object mapping selected UUIDs to "
        "their names; return {} if none is suitable.\n"
        + "question: "
        + question
        + "\ntopic entities:\n"
        + json.dumps(dict(seed_nodes), ensure_ascii=False)
        + "\nOutput:"
    )


def openrouter_topic_pruner(
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 512,
) -> Callable[[str], str]:
    """Create the LLM callable used by ``prune_topic_entities``."""
    from openai import OpenAI

    key = api_key or os.environ.get("LLM_API_KEY", "")
    if not key:
        raise ValueError("Set LLM_API_KEY before running topic pruning")
    client = OpenAI(
        api_key=key,
        base_url=base_url or os.environ.get("LLM_API_BASE", "https://openrouter.ai/api/v1"),
    )
    model_name = model or os.environ.get("LLM_MODEL", "google/gemma-4-31b-it")

    def generate(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return a JSON object of selected DSM graph topic entities."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return generate


def prune_topic_entities(
    question: str,
    seed_nodes: Mapping[str, str],
    *,
    llm_fn: Callable[[str], str] | None = None,
    width: int = 5,
    max_seed_nodes: int = 10,
) -> dict[str, str]:
    """Return a subset of the supplied ``{UUID: name}`` seed nodes.

    Pruning runs when the number of seed nodes exceeds ``width``. A valid
    empty JSON object means the LLM selected no suitable topic entity.
    Malformed output falls back to the first ``width`` input nodes so the
    configured width is never exceeded.
    """
    if width < 1 or max_seed_nodes < width:
        raise ValueError("Require 1 <= width <= max_seed_nodes")
    seeds = dict(list(seed_nodes.items())[:max_seed_nodes])
    if len(seeds) <= width:
        return seeds

    generate = llm_fn or openrouter_topic_pruner()
    response = generate(build_topic_prune_prompt(question, seeds, width=width))
    try:
        selected = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        return dict(list(seeds.items())[:width])
    if not isinstance(selected, dict):
        return dict(list(seeds.items())[:width])
    if not selected:
        return {}

    # Accept UUIDs only from the supplied seed mapping. Canonical names come
    # from the graph, never from free-form LLM text.
    chosen = {
        entity_id: name
        for entity_id, name in seeds.items()
        if entity_id in selected
    }
    return dict(list(chosen.items())[:width]) if chosen else dict(list(seeds.items())[:width])
