"""Stage 2: extract structured fields (environment, frequency, money impact) from bug reports."""
from __future__ import annotations

import re

from pydantic import BaseModel

import rules
from llm import FAST_MODEL, LLM, chunks
from models import BugFields, Message


ENV_TAIL = re.compile(r"(\.\s*phone:.*|\s+version \d.*|\s+(android|ios|ipados)\s?\d[\d.]*$)", re.I)


def _device(d):
    """Normalize casing so 'samsung s23' and 'Samsung S23' count as one device."""
    if not d:
        return None
    words = []
    for w in d.split():
        lw = w.lower()
        words.append({"iphone": "iPhone", "ipad": "iPad", "pro": "Pro"}.get(lw, w[:1].upper() + w[1:]))
    return " ".join(words)


def _first(regex, text):
    m = regex.search(text)
    if not m:
        return None
    return next((g for g in m.groups() if g), m.group(0)) if m.groups() else m.group(0)


def extract_rule(msg: Message) -> BugFields:
    t = msg.text
    low = t.lower()
    platform = next((p for rx, p in rules.PLATFORM_RE if rx.search(t)), None)
    os_ver = _first(rules.OS_RE, t)
    if os_ver:
        name, _, num = os_ver.lower().replace("os", "os ").partition(" ")
        name = {"android": "Android", "ios": "iOS", "ipados": "iPadOS"}.get(name.strip(), name)
        os_ver = f"{name} {num.strip()}".strip()
    if os_ver and not platform:
        platform = "iOS" if "os" in os_ver.lower() and "android" not in os_ver.lower() else "Android"
    freq = None
    if any(w in low for w in ["every time", "always", "everytime", "instantly", "keep", "every ",
                              "sürekli", "immediatly", "immediately", "in a row"]):
        freq = "always"
    elif any(w in low for w in ["sometimes", "randomly", "bazen"]):
        freq = "sometimes"
    summary = ENV_TAIL.sub("", t.split(" - ")[0].split(" (")[0]).strip(" .")
    summary = summary[:1].upper() + summary[1:]
    return BugFields(
        id=msg.id,
        summary=summary[:120],
        platform=platform,
        os_version=os_ver,
        device=_device(_first(rules.DEVICE_RE, t)),
        game_version=_first(rules.GAME_VERSION_RE, t),
        steps_hint=None,
        frequency=freq,
        monetization_impact=rules.has_any(t, rules.CATEGORY_KEYWORDS["payment"]) > 0,
        has_screenshot=rules.has_any(t, rules.SCREENSHOT) > 0,
        language="tr" if rules.is_turkish(t) else "en",
    )


SYSTEM = """You extract structured fields from player bug reports for "Neon Drift", a mobile racing game.
For each report fill only what the player actually said; use null when not stated. Never invent devices,
OS versions or game versions.
- summary: one neutral English line describing the problem (translate Turkish).
- platform: Android or iOS if stated or clearly implied by device/store (Google Play = Android, Apple = iOS).
- os_version e.g. "Android 14", "iOS 17.5". game_version e.g. "2.4.1".
- steps_hint: what the player was doing when it happened, if stated.
- frequency: always / sometimes / once, only if indicated.
- monetization_impact: true if real money, purchases, coin packs or subscriptions were lost or charged.
- has_screenshot: true if the player says they attached a screenshot/receipt.
- language: "tr" if the message is Turkish, else "en"."""


class ExtractionBatch(BaseModel):
    items: list[BugFields]


def extract_ai(messages: list[Message], llm: LLM, batch_size: int = 20) -> list[BugFields]:
    out: dict[str, BugFields] = {}
    for batch in chunks(messages, batch_size):
        user = "\n".join(f"[{m.id}] {m.text}" for m in batch)
        for item in llm.parse(FAST_MODEL, SYSTEM, user, ExtractionBatch, max_tokens=8000).items:
            out[item.id] = item
    return [out.get(m.id) or extract_rule(m) for m in messages]


def extract(messages: list[Message], llm: LLM | None = None) -> list[BugFields]:
    return extract_ai(messages, llm) if llm else [extract_rule(m) for m in messages]
