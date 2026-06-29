"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from typing import Any


def _model_name(override: str | None, default: str) -> str:
    return override or os.getenv("LLM_MODEL") or default


def get_llm(model: str | None = None, temperature: float = 0.0) -> Any:
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. FIREWORKS_API_KEY → ChatOpenAI with Fireworks OpenAI-compatible endpoint
    2. GEMINI_API_KEY → ChatGoogleGenerativeAI
    3. OPENAI_API_KEY → ChatOpenAI
    4. ANTHROPIC_API_KEY → ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    if os.getenv("FIREWORKS_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        fireworks_config: dict[str, Any] = {
            "model": _model_name(model, "accounts/fireworks/models/deepseek-v4-pro"),
            "api_key": os.environ["FIREWORKS_API_KEY"],
            "base_url": os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
            "temperature": temperature,
            "extra_body": {
                "top_k": int(os.getenv("FIREWORKS_TOP_K", "40")),
            },
        }
        max_tokens = os.getenv("FIREWORKS_MAX_TOKENS")
        if max_tokens:
            fireworks_config["max_tokens"] = int(max_tokens)
        return ChatOpenAI(**fireworks_config)

    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=_model_name(model, "gemini-2.5-flash"),
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=_model_name(model, "gpt-4o-mini"),
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=_model_name(model, "claude-sonnet-4-20250514"),
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set FIREWORKS_API_KEY, GEMINI_API_KEY, "
        "OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
