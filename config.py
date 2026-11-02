"""
Central settings, loaded from environment variables (see .env).
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _clean_env(key: str, default: str = "") -> str:
    """Strip whitespace/newlines AND accidental surrounding quote
    characters — e.g. CF_API_TOKEN="abc123" in .env would otherwise load
    as the literal string '"abc123"', quotes included, which breaks
    Bearer-token auth silently (no error until the API rejects it)."""
    value = os.environ.get(key, default).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value


def _clean_int_env(key: str, default: int) -> int:
    """Int env var with a real numeric default, so a missing/blank var
    falls back to `default` instead of crashing on int("")."""
    raw = _clean_env(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key}={raw!r} is not a valid integer") from exc


# --- Qdrant ---
QDRANT_HOST = _clean_env("QDRANT_HOST", "localhost")
QDRANT_PORT = _clean_int_env("QDRANT_PORT", 6333)
QDRANT_TIMEOUT = _clean_int_env("QDRANT_TIMEOUT", 30)
COLLECTION_NAME = _clean_env("COLLECTION_NAME", "wolf359_scripts")

# --- Jina (embeddings + rerank) ---
JINA_API_KEY = _clean_env("JINA_API_KEY", "")
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
EMBED_MODEL = _clean_env("EMBED_MODEL", "jina-embeddings-v3")
EMBED_DIM = _clean_int_env("EMBED_DIM", 1024)
RERANK_MODEL = _clean_env("RERANK_MODEL", "jina-reranker-v2-base-multilingual")

# --- Cloudflare Workers AI (generation) ---
CF_ACCOUNT_ID = _clean_env("CF_ACCOUNT_ID", "")
CF_API_TOKEN = _clean_env("CF_API_TOKEN", "")
# NOTE: confirm this model name is actually available on your Cloudflare
# account before relying on it — Workers AI's model catalog changes, and
# this default is a best guess, not a verified-current value.
CF_MODEL = _clean_env("CF_MODEL", "")
CF_GENERATION_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"
)


# --- Retrieval tuning ---
FETCH_K = _clean_int_env("FETCH_K", 10)   # candidates pulled from Qdrant before rerank
TOP_K = _clean_int_env("TOP_K", 5)
QUERY_EXPANSION_VARIANTS = _clean_int_env("QUERY_EXPANSION_VARIANTS", 2)
SCENE_WINDOW = _clean_int_env("SCENE_WINDOW", 2)  # neighbors on each side of a matched chunk
RERANK_SCOPE = _clean_env("RERANK_SCOPE", "chunk").lower()
WINDOW_RERANK_POOL = _clean_int_env("WINDOW_RERANK_POOL", 8)

# --- Conversation history ---
# Window = 5 "turns". A summarizer turn = 1 user + 1 assistant message.
# A group-chat turn = 1 user + 1-3 character replies. Both are stored
# as a flat message list per session; get_history() trims by message
# count appropriate to the mode (see core/history.py).
HISTORY_WINDOW_TURNS = _clean_int_env("HISTORY_WINDOW_TURNS", 5)

# --- Token usage ---
# Hard cap on each model response. The session budget is a soft warning
# threshold only: requests are never blocked, the UI just turns red once
# the cumulative input+output token count reaches it.
MAX_OUTPUT_TOKENS = _clean_int_env("MAX_OUTPUT_TOKENS", 1024)
SESSION_TOKEN_BUDGET = _clean_int_env("SESSION_TOKEN_BUDGET", 25000)

GROUP_CHAT_CHARACTERS = ["MINKOWSKI", "EIFFEL", "HILBERT"]


def validate() -> None:
    """Fail fast when required settings are missing or inconsistent.

    Called from the FastAPI lifespan hook so misconfiguration surfaces at
    startup instead of on the first user request.
    """
    required = {
        "JINA_API_KEY": JINA_API_KEY,
        "CF_ACCOUNT_ID": CF_ACCOUNT_ID,
        "CF_API_TOKEN": CF_API_TOKEN,
        "CF_MODEL": CF_MODEL,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in the values."
        )

    if QDRANT_PORT <= 0:
        raise ValueError("QDRANT_PORT must be a positive integer")
    if EMBED_DIM <= 0:
        raise ValueError("EMBED_DIM must be a positive integer")
    if FETCH_K <= 0:
        raise ValueError("FETCH_K must be a positive integer")
    if not 0 < TOP_K <= FETCH_K:
        raise ValueError("TOP_K must be between 1 and FETCH_K")
    if QUERY_EXPANSION_VARIANTS < 0:
        raise ValueError("QUERY_EXPANSION_VARIANTS must be >= 0")
    if SCENE_WINDOW < 0:
        raise ValueError("SCENE_WINDOW must be >= 0")
    if RERANK_SCOPE not in {"chunk", "window"}:
        raise ValueError("RERANK_SCOPE must be 'chunk' or 'window'")
    if WINDOW_RERANK_POOL <= 0:
        raise ValueError("WINDOW_RERANK_POOL must be a positive integer")
    if HISTORY_WINDOW_TURNS < 1:
        raise ValueError("HISTORY_WINDOW_TURNS must be >= 1")
    if MAX_OUTPUT_TOKENS <= 0:
        raise ValueError("MAX_OUTPUT_TOKENS must be a positive integer")
    if SESSION_TOKEN_BUDGET <= 0:
        raise ValueError("SESSION_TOKEN_BUDGET must be a positive integer")

    try:
        from core import personas
    except Exception:
        return
    missing_personas = [
        name for name in GROUP_CHAT_CHARACTERS if name not in personas.SYSTEM_PROMPTS
    ]
    if missing_personas:
        raise ValueError(
            "GROUP_CHAT_CHARACTERS lists character(s) with no persona prompt: "
            + ", ".join(missing_personas)
        )
