"""
Shared Jina embedding client.

Used by both query-time retrieval (task="retrieval.query") and offline
ingestion (task="retrieval.passage") so the two paths can never drift.
"""
import requests

import config

QUERY_TASK = "retrieval.query"
PASSAGE_TASK = "retrieval.passage"


def embed(
    texts: list[str],
    task: str = QUERY_TASK,
    batch_size: int = 64,
    timeout: int = 60,
) -> list[list[float]]:
    """Embed texts in batches, preserving input order."""
    if not texts:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = requests.post(
            config.JINA_EMBED_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.JINA_API_KEY}",
            },
            json={
                "model": config.EMBED_MODEL,
                "task": task,
                "normalized": True,
                "input": batch,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors.extend(item["embedding"] for item in ordered)
    return vectors


def embed_query(query: str, timeout: int = 60) -> list[float]:
    return embed([query], task=QUERY_TASK, timeout=timeout)[0]
