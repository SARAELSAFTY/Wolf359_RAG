"""
Group chat: on each user message, a random 1-3 of the characters
(Minkowski, Eiffel, Hilbert) reply once, in a randomly shuffled order.
If the user directly addresses one or more characters by name (e.g.
"Hilbert, what do you think?" or "@Hilbert"), only the addressed
character(s) reply. Each one can see what the earlier characters in THIS
round already said (so they can react to each other, not just to the
user). Everything — the user message and all replies — goes into one
shared history.
"""
import random
import re

import config
from core import retrieval, generation, history, personas

MESSAGES_PER_TURN = 1 + len(config.GROUP_CHAT_CHARACTERS)  # max: 1 user + 3 characters

ADDRESS_ALIASES = {
    "MINKOWSKI": ("minkowski", "commander minkowski", "commander", "renee", "renée"),
    "EIFFEL": ("eiffel", "doug eiffel", "doug"),
    "HILBERT": ("hilbert", "dr. hilbert", "dr hilbert", "doctor hilbert"),
}

def _addressed_characters(query: str) -> list[str]:
    """Return characters the user is directly addressing, in query order.

    Only vocative-style mentions count — "Hilbert, ..." / "Hey Hilbert" /
    "@Hilbert" / "... , Hilbert?" — so a question that merely mentions a
    character ("What did Hilbert do to Eiffel?") still gets the normal
    random group response.
    """
    text = query.strip().lower()
    positions = {}

    for character, aliases in ADDRESS_ALIASES.items():
        for alias in aliases:
            escaped = re.escape(alias)
            patterns = [
                rf"@{escaped}(?!\w)",
                rf"^\s*{escaped}\s*[,:;!?]",
                rf"^\s*{escaped}\s*$",
                rf"(?:^|[,;.!?]\s+){escaped}\s*[,;:!?]",
                rf",\s*{escaped}\s*[.!?]*\s*$",
                rf"^\s*{escaped}\s+(?:please\s+)?(?:tell|explain|describe|say|respond|answer|help)\b",
            ]
            if text.endswith("?"):
                patterns.append(
                    rf"^\s*{escaped}\s+(?:please\s+)?(?:what|why|how|who|when|where|do|can|could|would|will)\b"
                )
            matches = [m.start() for p in patterns for m in re.finditer(p, text)]
            if matches:
                positions[character] = min(matches)
                break

    return [c for c, _ in sorted(positions.items(), key=lambda item: item[1])]


def chat(query: str, session_id: str) -> dict:
    # Scene-level, unfiltered retrieval — group chat is about the
    # conversation/topic generally, not one character's lines specifically.
    past_turns = history.get_history(
        session_id, max_messages=config.HISTORY_WINDOW_TURNS * MESSAGES_PER_TURN
    )
    results = retrieval.search(query, character=None, session_id=session_id)
    context_block = retrieval.build_context_block(results) if results else "(no closely matching script excerpts found)"

    addressed = [
        character
        for character in _addressed_characters(query)
        if character in config.GROUP_CHAT_CHARACTERS
        and character in personas.SYSTEM_PROMPTS
    ]
    if addressed:
        order = addressed
    else:
        speaker_count = random.randint(1, len(config.GROUP_CHAT_CHARACTERS))
        order = random.sample(config.GROUP_CHAT_CHARACTERS, speaker_count)

    new_messages = [{"role": "user", "content": query}]
    replies = []

    for character_name in order:
        system_prompt = personas.SYSTEM_PROMPTS[character_name]
        messages = [{"role": "system", "content": f"{system_prompt}\n\nContext from the scripts:\n{context_block}"}]
        messages.extend(past_turns)
        # include this round's replies-so-far, so later characters can react
        messages.extend(new_messages)
        messages.append({
            "role": "user",
            "content": f"(Respond in character as {character_name} now.)",
        })

        reply_text = generation.generate(messages, session_id=session_id).text
        replies.append({"character": character_name, "text": reply_text})
        new_messages.append({"role": "assistant", "content": f"{character_name}: {reply_text}"})

    history.add_messages(session_id, new_messages)

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
    return {"replies": replies, "sources": sources}
