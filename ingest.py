"""
Offline ingestion for the Wolf 359 scripts.

Reads JSONL records shaped like the existing Qdrant payload:

    {"text": "...", "chunk_index": 13, "chunk_ID": "<uuid>",
     "episode": 36, "title": "Fire and Brimstone", "character name": "KEPLER"}

`character name` intentionally keeps the space — that is the real payload
key used by core/retrieval.py and the live collection.

Usage:
    python -m ingest --input data/scripts.jsonl --dry-run
    python -m ingest --input data/scripts.jsonl
    python -m ingest --input data/scripts.jsonl --recreate
"""
import argparse
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from qdrant_client.models import Distance, PointStruct, VectorParams

import config
from core.embeddings import PASSAGE_TASK, embed
from core.retrieval import get_qdrant_client

CHARACTER_FIELD = "character name"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            records.append(record)
    return records


def _require_string(record: dict, key: str, position: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record {position}: missing non-empty string {key!r}")
    return value.strip()


def _to_int(value, key: str, position: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {position}: {key!r} must be an integer") from exc


def _stable_chunk_id(episode: int, title: str, chunk_index: int, text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{episode}|{title}|{chunk_index}|{text}"))


def normalize_records(records: list[dict]) -> list[dict]:
    """Validate records and return Qdrant-ready {id, payload} dicts.

    Missing chunk_index values are assigned sequentially per (episode,
    title). Missing chunk_ID values get a deterministic uuid5 so re-running
    ingestion is idempotent.
    """
    points = []
    next_index: dict[tuple[int, str], int] = defaultdict(int)
    seen_ids: dict[str, int] = {}
    seen_chunk_positions: dict[tuple[int, str, int], int] = {}

    for position, record in enumerate(records):
        text = _require_string(record, "text", position)
        character = _require_string(record, CHARACTER_FIELD, position)
        title = _require_string(record, "title", position)
        episode = _to_int(record.get("episode"), "episode", position)

        group = (episode, title)
        if record.get("chunk_index") is None:
            chunk_index = next_index[group]
        else:
            chunk_index = _to_int(record["chunk_index"], "chunk_index", position)
        next_index[group] = max(next_index[group], chunk_index + 1)

        chunk_key = (episode, title, chunk_index)
        if chunk_key in seen_chunk_positions:
            raise ValueError(
                f"record {position}: duplicate chunk_index {chunk_index} for "
                f"episode {episode!r} / title {title!r} "
                f"(first seen at record {seen_chunk_positions[chunk_key]})"
            )
        seen_chunk_positions[chunk_key] = position

        raw_id = record.get("chunk_ID")
        if raw_id is None or not str(raw_id).strip():
            chunk_id = _stable_chunk_id(episode, title, chunk_index, text)
        else:
            chunk_id = str(raw_id).strip()
            try:
                uuid.UUID(chunk_id)
            except ValueError as exc:
                raise ValueError(
                    f"record {position}: chunk_ID {chunk_id!r} is not a valid UUID"
                ) from exc

        if chunk_id in seen_ids:
            raise ValueError(
                f"record {position}: duplicate chunk_ID {chunk_id!r} "
                f"(first seen at record {seen_ids[chunk_id]})"
            )
        seen_ids[chunk_id] = position

        points.append({
            "id": chunk_id,
            "payload": {
                "text": text,
                "chunk_index": chunk_index,
                "chunk_ID": chunk_id,
                "episode": episode,
                "title": title,
                CHARACTER_FIELD: character,
            },
        })

    return points


def _collection_size(client) -> int | None:
    info = client.get_collection(config.COLLECTION_NAME)
    vectors = info.config.params.vectors
    return next(iter(vectors.values())).size if isinstance(vectors, dict) else vectors.size


def ensure_collection(client, recreate: bool) -> None:
    exists = client.collection_exists(config.COLLECTION_NAME)
    if recreate and exists:
        client.delete_collection(config.COLLECTION_NAME)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )
        return

    size = _collection_size(client)
    if size != config.EMBED_DIM:
        raise RuntimeError(
            f"Collection {config.COLLECTION_NAME!r} has vector size {size}, "
            f"but EMBED_DIM is {config.EMBED_DIM}. Use --recreate to rebuild it."
        )


def upsert_points(client, points: list[dict], batch_size: int) -> None:
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        vectors = embed(
            [point["payload"]["text"] for point in batch],
            task=PASSAGE_TASK,
            batch_size=batch_size,
        )
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embed() returned {len(vectors)} vector(s) for a batch of "
                f"{len(batch)} text(s) starting at offset {start}; refusing to "
                "upsert a mismatched/truncated batch."
            )
        structs = [
            PointStruct(id=point["id"], vector=vector, payload=point["payload"])
            for point, vector in zip(batch, vectors)
        ]
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=structs,
            wait=True,
        )
        print(f"  upserted {min(start + batch_size, len(points))}/{len(points)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Wolf 359 script JSONL into Qdrant.")
    parser.add_argument("--input", required=True, help="Path to a JSONL file")
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the collection")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding/upsert batch size")
    parser.add_argument("--dry-run", action="store_true", help="Validate input without network calls")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        print("error: --batch-size must be positive", file=sys.stderr)
        return 2

    try:
        records = load_jsonl(Path(args.input))
        points = normalize_records(records)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    episodes = {(p["payload"]["episode"], p["payload"]["title"]) for p in points}
    characters = sorted({p["payload"][CHARACTER_FIELD] for p in points})
    print(
        f"Loaded {len(points)} chunks across {len(episodes)} episode(s) "
        f"and {len(characters)} character(s)."
    )

    if args.dry_run:
        print("Dry run: input validated; no embeddings or writes performed.")
        return 0

    try:
        client = get_qdrant_client()
        ensure_collection(client, recreate=args.recreate)
        upsert_points(client, points, batch_size=args.batch_size)
    except Exception as exc:
        print(f"error: ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Collection {config.COLLECTION_NAME!r} now contains the ingested chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
