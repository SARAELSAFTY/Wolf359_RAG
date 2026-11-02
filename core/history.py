"""
In-memory conversation history, keyed by session_id.

Stored as a flat, unbounded list of {"role", "content"} messages per
session; trimming to a "window" happens at READ time via max_messages,
since summarizer and group-chat turns have different shapes:
  - summarizer turn = 1 user + 1 assistant message  (2 messages/turn)
  - group-chat turn = 1 user + 1-3 character replies (up to 4 messages/turn)
So "window = 5 turns" means a different max_messages depending on mode —
see HISTORY_WINDOW_TURNS usage in features/summarizer.py and
features/groupchat.py.

Same caveats as before: in-memory only (lost on restart, not safe across
multiple worker processes), sessions never expire. Fine for local/single-
process use; revisit before any real multi-user deployment.
"""
from collections import defaultdict

import config

Message = dict  # {"role": "user" | "assistant", "content": str}

_history: dict[str, list[Message]] = defaultdict(list)


def get_history(session_id: str, max_messages: int) -> list[Message]:
    return _history[session_id][-max_messages:]


def add_messages(session_id: str, messages: list[Message]) -> None:
    _history[session_id].extend(messages)


def clear_history(session_id: str) -> None:
    _history.pop(session_id, None)
