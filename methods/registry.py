from __future__ import annotations

from typing import Any


def available_methods() -> tuple[str, ...]:
    return ("hipporag2",)


def create_method(name: str, **kwargs: Any):
    normalized = name.casefold().replace("-", "").replace("_", "")
    if normalized == "hipporag2":
        from .hipporag import HippoRAG2Method

        return HippoRAG2Method(**kwargs)
    raise ValueError(f"Unknown method {name!r}; available methods: {', '.join(available_methods())}")
