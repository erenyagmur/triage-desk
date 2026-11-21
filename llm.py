"""Thin wrapper around the official Anthropic SDK with structured outputs and a disk cache.

Every AI stage calls `LLM.parse(...)`. Tests pass a fake object with the same method,
so the real API is never hit in the test suite.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

FAST_MODEL = "claude-haiku-5-5"     # per-message classification + extraction
SMART_MODEL = "claude-sonnet-5-5"   # grouping, severity reasoning, ticket drafting

T = TypeVar("T", bound=BaseModel)
CACHE_DIR = Path(__file__).parent / "out" / ".llm_cache"


class LLMError(RuntimeError):
    pass


def ai_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class LLM:
    def __init__(self, client=None, cache: bool = True):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.cache = cache
        self.calls = 0

    def parse(self, model: str, system: str, user: str, schema: Type[T], max_tokens: int = 16000) -> T:
        key = hashlib.sha256(json.dumps([model, system, user, schema.__name__]).encode()).hexdigest()[:24]
        path = CACHE_DIR / f"{key}.json"
        if self.cache and path.exists():
            return schema.model_validate_json(path.read_text(encoding="utf-8"))

        kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                      messages=[{"role": "user", "content": user}], output_format=schema)
        if model == SMART_MODEL:
            # Server-side fallback if a safety classifier declines (routes by refusal category).
            response = self.client.beta.messages.parse(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            response = self.client.messages.parse(**kwargs)
        self.calls += 1

        if response.stop_reason == "refusal":
            raise LLMError(f"{model} declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{model} hit max_tokens; output truncated")
        parsed: Optional[T] = response.parsed_output
        if parsed is None:
            raise LLMError(f"{model} returned no parseable output")
        if self.cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(parsed.model_dump_json(), encoding="utf-8")
        return parsed


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
