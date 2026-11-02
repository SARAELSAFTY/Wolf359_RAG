import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # must run before `import config` reads any environment variables

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from schemas import (
    SummarizeRequest, SummarizeResponse,
    GroupChatRequest, GroupChatResponse,
    SourceChunk, ClearSessionsRequest, TokenUsage,
)
from features import summarizer, groupchat
from core import history, token_usage, retrieval

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.validate()
    try:
        retrieval.verify_collection()
        logger.info("Connected to Qdrant collection %s", config.COLLECTION_NAME)
    except Exception as exc:
        logger.warning(
            "Qdrant collection check failed for %s: %s",
            config.COLLECTION_NAME,
            exc,
        )
    yield


app = FastAPI(title="Wolf 359 RAG", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


def _token_usage(session_id: str) -> TokenUsage:
    used = token_usage.get(session_id)
    limit = config.SESSION_TOKEN_BUDGET
    return TokenUsage(
        session_tokens=used,
        token_limit=limit,
        over_limit=used >= limit,
    )


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    try:
        result = summarizer.summarize(req.query, req.session_id, character=req.character)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SummarizeResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        session_id=req.session_id,
        token_usage=_token_usage(req.session_id),
    )


@app.post("/groupchat", response_model=GroupChatResponse)
def group_chat(req: GroupChatRequest):
    try:
        result = groupchat.chat(req.query, req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GroupChatResponse(
        replies=result["replies"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        session_id=req.session_id,
        token_usage=_token_usage(req.session_id),
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    history.clear_history(session_id)
    token_usage.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/session/clear")
def clear_sessions(req: ClearSessionsRequest):
    for session_id in req.session_ids:
        history.clear_history(session_id)
        token_usage.clear(session_id)
    return {"status": "cleared", "session_ids": req.session_ids}
