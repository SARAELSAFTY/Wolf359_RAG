"""
Thin wrapper around Cloudflare Workers AI's chat-completion endpoint.
URL structure and CF_MODEL default confirmed against Cloudflare's current
docs during setup.
"""
from dataclasses import dataclass

import requests

import config
from core import token_usage


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _estimate_tokens(text: str) -> int:
    # Rough fallback for models/endpoints that don't return a usage block.
    return max(1, len(text) // 4) if text else 0


def _estimate_message_tokens(messages: list[dict]) -> int:
    return sum(_estimate_tokens(str(m.get("content", ""))) for m in messages)


def generate(
    messages: list[dict],
    max_tokens: int | None = None,
    session_id: str | None = None,
) -> GenerationResult:
    if max_tokens is None:
        max_tokens = config.MAX_OUTPUT_TOKENS

    payload = {"messages": messages}
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens

    response = requests.post(
        config.CF_GENERATION_URL,
        headers={"Authorization": f"Bearer {config.CF_API_TOKEN}"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    if not data.get("success", False):
        raise RuntimeError(f"Cloudflare Workers AI call failed: {data.get('errors')}")

    result = data["result"]
    text = result["response"]

    usage = result.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

    if total_tokens <= 0:
        prompt_tokens = prompt_tokens or _estimate_message_tokens(messages)
        completion_tokens = completion_tokens or _estimate_tokens(text)
        total_tokens = prompt_tokens + completion_tokens

    if session_id:
        token_usage.add(session_id, total_tokens)

    return GenerationResult(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
