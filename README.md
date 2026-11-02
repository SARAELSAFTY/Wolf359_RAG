# Wolf 359 RAG

RAG over the Wolf 359 audio drama scripts. Two modes:

- **Summarizer** — answers questions with cited script excerpts.
- **Group Chat** — Minkowski, Eiffel, and Hilbert reply in character.

Stack: FastAPI + Qdrant (vectors) + Jina (embeddings/rerank) + Cloudflare Workers AI (generation).

## Project layout

```
main.py                 FastAPI app and routes
config.py               Environment-backed settings + startup validation
schemas.py              Request/response models
ingest.py               JSONL -> Jina embeddings -> Qdrant
core/embeddings.py      Shared Jina embedding client
core/retrieval.py       Query expansion, Qdrant search, rerank, scene stitching
core/generation.py      Cloudflare Workers AI chat completion
core/history.py         In-memory conversation history
core/token_usage.py     In-memory per-session token counter
core/personas.py        Group-chat character prompts
features/               Summarizer and group-chat logic
eval_offline/           Live retrieval quality evaluation
static/                 Frontend
tests/                  Offline pytest suite (external services mocked)
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Create a local `.env` from the template and fill in the secrets:

```powershell
Copy-Item .env.example .env
```

`.env` is git-ignored. Commit `.env.example` only.

### Qdrant

The app expects a collection named by `COLLECTION_NAME` (default
`wolf359_scripts`) with a 1024-dimension cosine vector config. Run Qdrant
locally with Docker if needed:

```powershell
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Ingestion

Skip this section if your `wolf359_scripts` collection is already populated —
the app only reads from Qdrant. Ingestion exists for rebuilding the collection
or loading a new dataset.

Input is JSONL, one chunk per line. This is the same payload shape used by
the existing collection:

```json
{"text":"But Doctor Maxwell's in a situation.","chunk_index":13,"chunk_ID":"0002b6ef-420f-df22-e835-b68cda1e3d3d","episode":36,"title":"Fire and Brimstone","character name":"KEPLER"}
```

Required fields: `text`, `episode`, `title`, `character name`.
Optional: `chunk_index` (assigned sequentially per episode/title when
missing) and `chunk_ID` (a UUID; generated deterministically when missing).

```powershell
# Validate input without any network calls
python -m ingest --input data/scripts.jsonl --dry-run

# Embed and upsert (creates the collection if missing)
python -m ingest --input data/scripts.jsonl

# Drop and rebuild the collection first
python -m ingest --input data/scripts.jsonl --recreate
```

The `character name` key intentionally contains a space — that is the key
the retrieval code and the live collection use.

## Run

```powershell
uvicorn main:app --reload
```

Open http://localhost:8000. Routes: `/summarize`, `/groupchat`,
`/session/{session_id}`, `/session/clear`, `/health`.

Startup validates required environment variables and checks that the Qdrant
collection exists and matches `EMBED_DIM`.

## Tests

```powershell
python -m pytest
```

Tests are offline: Jina, Cloudflare, and Qdrant are mocked, so no credentials
or running services are needed.

## Retrieval tuning

Scene context is reconstructed by title, so minisodes/specials with no
`episode` value are stitched too. Set `RERANK_SCOPE=window` to rerank the
merged scene context instead of individual lines; the default is `chunk`.

## Retrieval eval (integration)

```powershell
python -m eval_offline.run_eval --min-hit-rate 0.75 --min-mrr 0.50 --min-context-hit-rate 0.75
```

This needs live Qdrant + Jina + Cloudflare access and exits non-zero when
quality falls below the thresholds. The report includes hit rate, MRR,
context hit rate (expected character present in the stitched scene), and
nDCG@k.
