"""Shared DSPy runtime for every method-level LLM call."""

from __future__ import annotations

from config.settings import Settings


def build_lm():
    """Create the configured DSPy LM without configuring a process-global client."""
    import dspy

    settings = Settings()
    return dspy.LM(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        temperature=0.0,
        max_retries=settings.llm_retry_attempts,
    )


def token_usage(lm) -> tuple[int, int]:
    """Best-effort usage extraction across DSPy/LiteLLM response versions."""
    history = getattr(lm, "history", None) or []
    if not history:
        return 0, 0
    usage = history[-1].get("usage") or {}
    return (
        int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
    )
