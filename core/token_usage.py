"""
In-memory cumulative token usage, keyed by session_id.

Same caveats as core/history.py: process-local only, lost on restart, and
not shared across multiple worker processes.
"""
from collections import defaultdict

_usage: dict[str, int] = defaultdict(int)


def add(session_id: str, tokens: int) -> None:
    if tokens > 0:
        _usage[session_id] += tokens


def get(session_id: str) -> int:
    return _usage.get(session_id, 0)


def clear(session_id: str) -> None:
    _usage.pop(session_id, None)
