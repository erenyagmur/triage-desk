"""Stage 1: classify each message as bug / feature_request / feedback / question / spam."""
from __future__ import annotations

import re

from pydantic import BaseModel

import rules
from llm import FAST_MODEL, LLM, chunks
from models import Classification, Message


def classify_rule(msg: Message) -> Classification:
    t = msg.text.lower().strip()
    bug_hits = sum(rules.has_any(t, kws) for kws in rules.CATEGORY_KEYWORDS.values())
    generic = rules.has_any(t, rules.GENERIC_BUG)

    def out(kind, conf):
        return Classification(id=msg.id, type=kind, confidence=conf)

    if rules.has_any(t, rules.SPAM) or len(re.sub(r"[^a-zçğışöü]", "", t)) < 4 or \
            re.fullmatch(r"(.)\1{4,}", t):
        return out("spam", 0.8)
    if bug_hits:
        return out("bug", min(0.95, 0.6 + 0.1 * (bug_hits + generic)))
    if any(t.startswith(q) for q in rules.QUESTION_START) or t.endswith("?"):
        return out("question", 0.7)
    if rules.has_any(t, rules.FEATURE):
        return out("feature_request", 0.7)
    if generic:
        return out("bug", 0.5)
    return out("feedback", 0.5)


SYSTEM = """You triage player messages from the Discord of "Neon Drift", a mobile racing game.
Classify each message into exactly one type:
- bug: the player reports something not working as intended. Vague reports ("game broken pls fix") count.
  Lost purchases, missing items, crashes, freezes, wrong UI text are bugs.
- feature_request: asks for something new or an option that does not exist.
- feedback: opinions, complaints about balance/prices/ads, praise. Not a malfunction.
- question: asks how something works or when something happens, no malfunction described.
- spam: ads, scams, self-promotion, off-topic chatter, emoji-only or meaningless messages.
Messages may be in English or Turkish and have typos. Give a confidence between 0 and 1.
Return one item per input message, using the given id."""


class ClassificationBatch(BaseModel):
    items: list[Classification]


def classify_ai(messages: list[Message], llm: LLM, batch_size: int = 25) -> list[Classification]:
    results: dict[str, Classification] = {}
    for batch in chunks(messages, batch_size):
        user = "\n".join(f"[{m.id}] {m.text}" for m in batch)
        parsed = llm.parse(FAST_MODEL, SYSTEM, user, ClassificationBatch, max_tokens=4000)
        for item in parsed.items:
            results[item.id] = item
    # Anything the model skipped falls back to rules rather than silently disappearing.
    return [results.get(m.id) or classify_rule(m) for m in messages]


def classify(messages: list[Message], llm: LLM | None = None) -> list[Classification]:
    return classify_ai(messages, llm) if llm else [classify_rule(m) for m in messages]
