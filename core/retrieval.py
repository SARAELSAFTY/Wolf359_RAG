"""
Retrieval: query expansion, vector search, rerank, and scene assembly.

The query-side pipeline stays the same in spirit as before: expand the
question, embed every variant, merge candidates, rerank, then reconstruct
scene context around each match. Scene lookup is title-based because titles
are unique in the corpus, which also supports minisodes/specials that have no
episode number.
"""
import json
import re
from collections import defaultdict

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, QueryRequest, Range

import config
from core import generation
from core.embeddings import QUERY_TASK, embed, embed_query

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=config.QDRANT_TIMEOUT
        )
    return _qdrant_client


NOISE_PATTERNS = [
    re.compile(r"\(MORE\)", re.IGNORECASE),
    re.compile(r"\bBEAT\.?", re.IGNORECASE),
    re.compile(r"\bSTATIC\.?", re.IGNORECASE),
    re.compile(r"^\d{1,4}\.?$"),
]


def clean_display_text(text: str) -> str:
    cleaned = text
    for pattern in NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" .")


def verify_collection() -> None:
    """Check the configured collection exists and matches EMBED_DIM."""
    info = get_qdrant_client().get_collection(config.COLLECTION_NAME)
    vectors = info.config.params.vectors
    size = next(iter(vectors.values())).size if isinstance(vectors, dict) else vectors.size
    if size != config.EMBED_DIM:
        raise RuntimeError(
            f"Collection {config.COLLECTION_NAME!r} has vector size {size}, "
            f"but EMBED_DIM is {config.EMBED_DIM}."
        )


def rerank(query: str, documents: list[str], top_n: int) -> list[dict]:
    response = requests.post(
        config.JINA_RERANK_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.JINA_API_KEY}"},
        json={"model": config.RERANK_MODEL, "query": query, "documents": documents, "top_n": top_n},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["results"]


def _normalize_variant(value, original: str) -> str | None:
    if not isinstance(value, str):
        return None
    variant = re.sub(r"\s+", " ", value.strip().strip("`").strip())
    if not variant or variant.casefold() == original.casefold():
        return None
    return variant[:400].rstrip()


def expand_query(
    query: str,
    n: int | None = None,
    session_id: str | None = None,
) -> list[str]:
    """Returns [original_query, variant_1, variant_2, ...]. Falls back to
    just [original_query] on any failure; expansion is a bonus, not a
    dependency."""
    if n is None:
        n = config.QUERY_EXPANSION_VARIANTS
    if n <= 0:
        return [query]

    prompt = (
        f"Reword the following question about the Wolf 359 audio drama in "
        f"{n} different ways, preserving its meaning but using different "
        f"vocabulary. Keep character names exactly as written. Respond with "
        f"ONLY a JSON array of {n} strings, nothing else — no explanation, "
        f"no markdown.\n\nQuestion: {query}"
    )
    try:
        raw = generation.generate(
            [{"role": "user", "content": prompt}], session_id=session_id
        ).text
        raw = raw.strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        variants = json.loads(raw)
        if isinstance(variants, list):
            result = [query]
            seen = {query.casefold()}
            for value in variants[:n]:
                variant = _normalize_variant(value, query)
                if variant and variant.casefold() not in seen:
                    seen.add(variant.casefold())
                    result.append(variant)
            return result
    except Exception:
        pass
    return [query]


def _embed_queries(queries: list[str]) -> list[list[float]]:
    try:
        return embed(queries, task=QUERY_TASK)
    except Exception:
        vectors = []
        for query in queries:
            try:
                vectors.append(embed_query(query))
            except Exception:
                vectors.append([])
        return vectors


def _query_variants(qdrant, vectors: list[list[float]], limit: int) -> list[list]:
    requests_payload = [
        QueryRequest(query=vector, limit=limit, with_payload=True) for vector in vectors
    ]
    try:
        responses = qdrant.query_batch_points(
            collection_name=config.COLLECTION_NAME, requests=requests_payload
        )
        return [response.points for response in responses]
    except Exception:
        batches = []
        for vector in vectors:
            try:
                batches.append(
                    qdrant.query_points(
                        collection_name=config.COLLECTION_NAME, query=vector, limit=limit
                    ).points
                )
            except Exception:
                batches.append([])
        return batches


def _scene_group(payload: dict):
    episode = payload.get("episode")
    title = payload.get("title")
    if episode is None and title is None:
        return None
    return (episode, title)


def _scene_id(group, lo: int, hi: int) -> str:
    episode, title = group
    return f"{title or ''}|{episode}|{lo}-{hi}"


def _single_neighbor(hit) -> dict:
    return {
        "character": hit.payload.get("character name"),
        "text": clean_display_text(hit.payload.get("text", "")),
        "chunk_index": hit.payload.get("chunk_index"),
    }


def _payload_to_neighbor(payload: dict) -> dict:
    return {
        "character": payload.get("character name"),
        "text": clean_display_text(payload.get("text", "")),
        "chunk_index": payload.get("chunk_index"),
    }


def _scroll_scene_range(group, lo: int, hi: int) -> list:
    episode, title = group
    if episode is None and title is None:
        return []

    must = [FieldCondition(key="chunk_index", range=Range(gte=lo, lte=hi))]
    if title is not None:
        must.append(FieldCondition(key="title", match=MatchValue(value=title)))
    if episode is not None:
        must.append(FieldCondition(key="episode", match=MatchValue(value=episode)))

    try:
        results, _ = get_qdrant_client().scroll(
            collection_name=config.COLLECTION_NAME,
            scroll_filter=Filter(must=must),
            limit=max(1, hi - lo + 1),
            with_payload=True,
        )
        return results
    except Exception:
        return []


def _fetch_scene_neighbors_batch(hits: list) -> dict:
    """Fetch neighbor windows once per title instead of once per hit."""
    by_hit = {}
    grouped = defaultdict(list)
    for hit in hits:
        group = _scene_group(hit.payload)
        chunk_index = hit.payload.get("chunk_index")
        if group is None or chunk_index is None:
            by_hit[hit.id] = [_single_neighbor(hit)]
        else:
            grouped[group].append(hit)

    for group, group_hits in grouped.items():
        indices = [hit.payload.get("chunk_index") for hit in group_hits]
        lo = min(indices) - config.SCENE_WINDOW
        hi = max(indices) + config.SCENE_WINDOW
        records = _scroll_scene_range(group, lo, hi)
        if not records:
            for hit in group_hits:
                by_hit[hit.id] = [_single_neighbor(hit)]
            continue

        neighbors = [_payload_to_neighbor(record.payload) for record in records]
        neighbors.sort(
            key=lambda item: item["chunk_index"]
            if isinstance(item["chunk_index"], int)
            else 10**9
        )
        for hit in group_hits:
            chunk_index = hit.payload.get("chunk_index")
            by_hit[hit.id] = [
                item
                for item in neighbors
                if isinstance(item["chunk_index"], int)
                and chunk_index - config.SCENE_WINDOW
                <= item["chunk_index"]
                <= chunk_index + config.SCENE_WINDOW
            ]
    return by_hit


def _format_scene_context(neighbors: list[dict]) -> str:
    lines = []
    for neighbor in neighbors:
        text = neighbor["text"]
        if not text:
            continue
        character = neighbor.get("character")
        if character:
            lines.append(f"{character}: {text}")
        else:
            lines.append(f"[stage direction] {text}")
    return "\n".join(lines)


def _build_scene_entries(selected: list[tuple], neighbors_by_hit: dict) -> list[dict]:
    """Return one entry per selected hit, with overlapping windows sharing a
    single merged scene context/scene_id."""
    records = []
    grouped = defaultdict(list)
    for index, (hit, text, score) in enumerate(selected):
        group = _scene_group(hit.payload)
        chunk_index = hit.payload.get("chunk_index")
        record = {
            "index": index,
            "hit": hit,
            "text": text,
            "score": score,
            "group": group,
            "chunk_index": chunk_index,
            "neighbors": neighbors_by_hit.get(hit.id, [_single_neighbor(hit)]),
        }
        records.append(record)
        if group is not None and chunk_index is not None:
            record["lo"] = chunk_index - config.SCENE_WINDOW
            record["hi"] = chunk_index + config.SCENE_WINDOW
            grouped[group].append(record)

    cluster_by_index = {}
    for group, group_records in grouped.items():
        group_records.sort(key=lambda item: (item["lo"], item["hi"], item["index"]))
        clusters = []
        for record in group_records:
            if not clusters or record["lo"] > clusters[-1]["hi"] + 1:
                clusters.append(
                    {
                        "group": group,
                        "lo": record["lo"],
                        "hi": record["hi"],
                        "records": [],
                    }
                )
            cluster = clusters[-1]
            cluster["records"].append(record)
            cluster["hi"] = max(cluster["hi"], record["hi"])
            cluster_by_index[record["index"]] = cluster

    context_by_cluster = {}
    for index, cluster in cluster_by_index.items():
        key = id(cluster)
        if key in context_by_cluster:
            continue
        merged = {}
        for record in cluster["records"]:
            for neighbor in record["neighbors"]:
                neighbor_index = neighbor.get("chunk_index")
                dedupe_key = (
                    neighbor_index
                    if neighbor_index is not None
                    else ("unknown", len(merged))
                )
                merged[dedupe_key] = neighbor
        ordered = sorted(
            merged.values(),
            key=lambda item: item["chunk_index"]
            if isinstance(item["chunk_index"], int)
            else 10**9,
        )
        context_by_cluster[key] = {
            "scene_id": _scene_id(cluster["group"], cluster["lo"], cluster["hi"]),
            "scene_context": _format_scene_context(ordered),
        }

    entries = []
    for record in records:
        cluster = cluster_by_index.get(record["index"])
        if cluster is None:
            entries.append(
                {
                    "hit": record["hit"],
                    "rerank_score": record["score"],
                    "cleaned_text": record["text"],
                    "scene_context": _format_scene_context(record["neighbors"])
                    or record["text"],
                    "scene_id": f"point:{record['hit'].id}",
                }
            )
            continue

        shared = context_by_cluster[id(cluster)]
        entries.append(
            {
                "hit": record["hit"],
                "rerank_score": record["score"],
                "cleaned_text": record["text"],
                "scene_context": shared["scene_context"] or record["text"],
                "scene_id": shared["scene_id"],
            }
        )
    return entries


def _dedupe_scenes(entries: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for entry in entries:
        if entry["scene_id"] in seen:
            continue
        seen.add(entry["scene_id"])
        deduped.append(entry)
    return deduped


def _rerank_candidates(query: str, candidates: list[tuple], top_n: int) -> list[tuple]:
    documents = [text for _, text in candidates]
    try:
        results = rerank(query, documents, top_n=min(top_n, len(documents)))
        selected = []
        for result in results:
            index = result["index"]
            if 0 <= index < len(candidates):
                hit, text = candidates[index]
                selected.append((hit, text, float(result["relevance_score"])))
        if selected:
            return selected
    except Exception:
        pass

    fallback = sorted(candidates, key=lambda item: item[0].score, reverse=True)
    return [
        (hit, text, float(hit.score))
        for hit, text in fallback[:top_n]
    ]


def _format_scene_heading(payload: dict) -> str:
    title = payload.get("title")
    episode = payload.get("episode")
    if title and episode is not None:
        return f"--- {title} (ep.{episode}) ---"
    if title:
        return f"--- {title} ---"
    if episode is not None:
        return f"--- Episode {episode} ---"
    return "--- Script excerpt ---"


def build_context_block(results: list[dict]) -> str:
    blocks = []
    seen = set()
    for result in results:
        key = result.get("scene_id") or result.get("scene_context")
        if key in seen:
            continue
        seen.add(key)
        payload = result["hit"].payload
        context = result.get("scene_context") or result.get("cleaned_text", "")
        blocks.append(f"{_format_scene_heading(payload)}\n{context}")
    return "\n\n".join(blocks)


def search(
    query: str,
    top_k: int | None = None,
    character: str | None = None,
    fetch_k: int | None = None,
    use_query_expansion: bool = True,
    use_scene_stitching: bool = True,
    session_id: str | None = None,
) -> list[dict]:
    """
    Returns a list of dicts, sorted by rerank relevance (highest first):
      - hit: the matched Qdrant point
      - rerank_score: float
      - cleaned_text: the matched chunk's text, noise-stripped
      - scene_context: the matched chunk PLUS its neighbors, stitched into
        a fuller scene block (same as cleaned_text if stitching is off)
      - scene_id: merged scene identity, used to deduplicate prompt context

    Character filtering remains post-retrieval: the payload key
    "character name" contains a space and cannot be used as a Qdrant filter
    path.
    """
    if top_k is None:
        top_k = config.TOP_K
    if fetch_k is None:
        fetch_k = config.FETCH_K

    qdrant = get_qdrant_client()
    queries = expand_query(query, session_id=session_id) if use_query_expansion else [query]

    vectors = [vector for vector in _embed_queries(queries) if vector]
    if not vectors:
        return []

    candidate_map = {}
    for points in _query_variants(qdrant, vectors, fetch_k):
        for hit in points:
            existing = candidate_map.get(hit.id)
            if existing is None or hit.score > existing.score:
                candidate_map[hit.id] = hit

    candidates = list(candidate_map.values())
    if character:
        candidates = [
            hit for hit in candidates if hit.payload.get("character name") == character
        ]
    if not candidates:
        return []

    ranked = []
    for hit in candidates:
        text = clean_display_text(hit.payload.get("text", ""))
        if text:
            ranked.append((hit, text))
    if not ranked:
        return []

    if use_scene_stitching and config.RERANK_SCOPE == "window":
        pool_size = max(top_k * 2, config.WINDOW_RERANK_POOL)
        first_pass = _rerank_candidates(query, ranked, pool_size)
        if not first_pass:
            return []

        hits = [hit for hit, _, _ in first_pass]
        neighbors = _fetch_scene_neighbors_batch(hits)
        scene_entries = _dedupe_scenes(_build_scene_entries(first_pass, neighbors))
        if not scene_entries:
            return []

        if len(scene_entries) > 1:
            try:
                results = rerank(
                    query,
                    [entry["scene_context"] for entry in scene_entries],
                    top_n=min(top_k, len(scene_entries)),
                )
                reranked = []
                for result in results:
                    index = result["index"]
                    if 0 <= index < len(scene_entries):
                        entry = dict(scene_entries[index])
                        entry["rerank_score"] = float(result["relevance_score"])
                        reranked.append(entry)
                if reranked:
                    return reranked
            except Exception:
                pass
        return scene_entries[:top_k]

    selected = _rerank_candidates(query, ranked, top_k)
    if not use_scene_stitching:
        return [
            {
                "hit": hit,
                "rerank_score": score,
                "cleaned_text": text,
                "scene_context": text,
                "scene_id": f"point:{hit.id}",
            }
            for hit, text, score in selected
        ]

    hits = [hit for hit, _, _ in selected]
    neighbors = _fetch_scene_neighbors_batch(hits)
    return _build_scene_entries(selected, neighbors)
