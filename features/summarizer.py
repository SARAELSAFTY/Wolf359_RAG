import re

import config
from core import retrieval, generation, history

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the Wolf 359 "
    "audio drama, using ONLY the script excerpts provided as context. "
    "If the provided excerpts don't contain enough information to answer "
    "confidently, say so plainly rather than guessing or inventing details. "
    "Cite which character said what when it's relevant to the answer."
)

LOW_CONFIDENCE_THRESHOLD = 0.3
MESSAGES_PER_TURN = 2  # 1 user + 1 assistant


def _strip_markdown(text: str) -> str:
    """Remove common markdown markers so the answer reads as plain text."""
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:[-*_])(?:\s*[-*_]){2,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*(?!\s)([^\n*]+?)(?<!\s)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(?!\s)([^\n_]+?)(?<!\s)_(?!\w)", r"\1", text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)
    text = text.replace("**", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def summarize(query: str, session_id: str, character: str | None = None) -> dict:
    results = retrieval.search(
        query,
        character=character,
        session_id=session_id,
    )

    if not results:
        answer = (
            "I couldn't find anything relevant in the scripts for that. "
            "Could you rephrase, or ask about something more specific?"
        )
        history.add_messages(session_id, [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ])
        return {"answer": answer, "sources": []}

    top_score = results[0]["rerank_score"]
    low_confidence_note = ""
    if top_score < LOW_CONFIDENCE_THRESHOLD:
        low_confidence_note = (
            "\n\n(Note: the retrieved excerpts had low relevance scores — "
            "this answer may be a weak match for your question.)"
        )

    context_block = retrieval.build_context_block(results)
    past_turns = history.get_history(session_id, max_messages=config.HISTORY_WINDOW_TURNS * MESSAGES_PER_TURN)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(past_turns)
    messages.append({
        "role": "user",
        "content": f"Context from the scripts:\n{context_block}\n\nQuestion: {query}",
    })

    answer = _strip_markdown(generation.generate(messages, session_id=session_id).text) + low_confidence_note
    history.add_messages(session_id, [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ])

    sources = [
        {
            "character": r["hit"].payload.get("character name"),
            "episode": r["hit"].payload.get("episode"),
            "title": r["hit"].payload.get("title"),
            "text": r["cleaned_text"],
            "rerank_score": r["rerank_score"],
        }
        for r in results
    ]
    return {"answer": answer, "sources": sources}
