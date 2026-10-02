"""Shared DSPy runtime for every method-level LLM call."""

from __future__ import annotations

from config.settings import Settings


def build_lm(*, temperature: float = 0.0):
    """Create the configured DSPy LM without configuring a process-global client."""
    import dspy

    settings = Settings()
    kwargs: dict = dict(
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        temperature=temperature,
        max_retries=settings.llm_retry_attempts,
    )
    # Optional OpenRouter provider routing (read from configs, not hard-coded)
    # Example: LLM_PROVIDER_ONLY=baidu/fp8, LLM_PROVIDER_ALLOW_FALLBACKS=false
    # maps to provider: {"only": ["baidu/fp8"], "allow_fallbacks": false}
    provider: dict = {}
    if settings.llm_provider_only:
        provider["only"] = [p.strip() for p in settings.llm_provider_only.split(",") if p.strip()]
    if settings.llm_provider_allow_fallbacks is not None:
        provider["allow_fallbacks"] = bool(settings.llm_provider_allow_fallbacks)
    if provider:
        # LiteLLM extra_body is forwarded to OpenRouter
        kwargs["extra_body"] = {"provider": provider}
    return dspy.LM(**kwargs)


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
