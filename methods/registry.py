from __future__ import annotations

from typing import Any


def available_methods() -> tuple[str, ...]:
    return ("hipporag2", "vanilla_rag", "fact_idf_rag", "cot", "tog", "tog2")


def create_method(name: str, **kwargs: Any):
    normalized = name.casefold().replace("-", "").replace("_", "")
    if normalized == "hipporag2":
        from .hipporag import HippoRAG2Method

        return HippoRAG2Method(**kwargs)
    if normalized == "vanillarag":
        from .vanilla_rag import VanillaRAGMethod

        return VanillaRAGMethod(**kwargs)
    if normalized == "factidfrag":
        from .hipporag import FactIDFRAGMethod

        return FactIDFRAGMethod(**kwargs)
    if normalized == "cot":
        from .cot.method import CoTMethod

        return CoTMethod(**kwargs)
    if normalized == "tog":
        from .tog import ToGMethod

        return ToGMethod(**kwargs)
    if normalized == "tog2":
        from .tog2 import ToG2Method

        return ToG2Method(**kwargs)
    raise ValueError(f"Unknown method {name!r}; available methods: {', '.join(available_methods())}")
