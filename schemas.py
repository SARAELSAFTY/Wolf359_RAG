from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    character: str | None
    episode: float | int | None
    title: str | None
    text: str
    rerank_score: float


class TokenUsage(BaseModel):
    session_tokens: int
    token_limit: int
    over_limit: bool


# --- Summarizer ---

class SummarizeRequest(BaseModel):
    session_id: str
    query: str
    character: str | None = Field(None, description="Optional: filter retrieval to one character's lines")


class SummarizeResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    session_id: str
    token_usage: TokenUsage


# --- Group chat ---

class GroupChatRequest(BaseModel):
    session_id: str
    query: str


class GroupChatReply(BaseModel):
    character: str
    text: str


class GroupChatResponse(BaseModel):
    replies: list[GroupChatReply]
    sources: list[SourceChunk]
    session_id: str
    token_usage: TokenUsage


# --- Session cleanup ---

class ClearSessionsRequest(BaseModel):
    session_ids: list[str]
