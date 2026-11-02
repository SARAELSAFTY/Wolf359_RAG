# Wolf 359 RAG

RAG over the Wolf 359 audio drama scripts. Two modes:

- **Summarizer** — answers questions with cited script excerpts.
- **Group Chat** — Minkowski, Eiffel, and Hilbert reply in character.

Stack: FastAPI + Qdrant (vectors, stored locally on this machine) + Jina (embeddings/rerank, hosted API) + Cloudflare Workers AI (generation, hosted API).

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

## Embeddings: stored locally, computed via Jina's API

The vectors that back retrieval live **on this machine**, not in a managed
cloud vector database. They sit inside the local Qdrant instance started by
the `docker run` command above, which persists to a Docker volume on disk.
Once ingestion has finished, searching the collection (`/summarize`,
`/groupchat`) never leaves your machine for the vector lookup itself — the
only outbound calls at query time are to Jina (query embedding + rerank) and
Cloudflare Workers AI (generation).

The embedding *values* themselves are not computed locally — there's no
local embedding model running — they're produced by calling Jina's hosted
embeddings endpoint once per chunk, at ingestion time, and then written to
the local Qdrant store. The chunks are the show's dialogue/script lines,
broken out one character line (or stage direction) per JSONL record.

### How the embeddings were made

1. **Source data**: the Wolf 359 script text was split into one JSONL record
   per line of dialogue/stage direction, each tagged with `episode`,
   `title`, and `character name` (see the payload shape below).
2. **Embedding call**: `ingest.py` batches those records' `text` fields and
   POSTs each batch to Jina's `/v1/embeddings` endpoint
   (`core/embeddings.py`), using model `jina-embeddings-v3`
   (`EMBED_MODEL`), task `retrieval.passage` (the asymmetric, passage-side
   embedding — queries use `retrieval.query` instead so the two stay
   compatible), and 1024-dimension output (`EMBED_DIM`). This is the one
   step that needs network access and a valid `JINA_API_KEY`.
3. **Local storage**: the returned vectors are packaged into Qdrant
   `PointStruct`s — vector + full payload (`text`, `chunk_index`,
   `chunk_ID`, `episode`, `title`, `character name`) — and upserted into
   the local Qdrant collection named by `COLLECTION_NAME`
   (default `wolf359_scripts`), created with a 1024-dim cosine distance
   config if it doesn't already exist.
4. **Idempotency**: `chunk_ID` is a deterministic UUID5 derived from
   `(episode, title, chunk_index, text)` when not supplied, so re-running
   ingestion against the same source JSONL re-upserts the same point IDs
   instead of duplicating them.

Re-running ingestion (see below) is the only way to regenerate or update
the local embeddings — nothing recomputes them automatically, and deleting
the Qdrant Docker volume deletes them for good.

## Ingestion

Skip this section if your `wolf359_scripts` collection is already populated —
the app only reads from Qdrant. Ingestion exists for rebuilding the local
collection or loading a new dataset.

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
