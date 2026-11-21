"""Stage 3: group duplicate bug reports into issues, with a confidence score and a representative."""
from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel, Field

import rules
from llm import LLM, SMART_MODEL
from models import BugFields, Issue, Message

ENV_NOISE = re.compile(
    r"(android\s?\d+|i(?:pad)?os\s?[\d.]+|v?\d+\.\d+(?:\.\d+)?|version|samsung|galaxy|pixel|xiaomi|redmi|"
    r"note|oppo|iphone|ipad|pro|gen|phone:|\b[as]\d{2}\b|screenshot attached|\d+)", re.I)
DISTANCE_THRESHOLD = 0.85   # cosine distance; swept 0.6-0.9 on the synthetic set (see README limitations)


def is_vague(msg: Message) -> bool:
    """A bug report with no concrete symptom ("game broken pls fix") can't be grouped meaningfully."""
    return not any(rules.has_any(msg.text, kws) for kws in rules.CATEGORY_KEYWORDS.values())


def representative(ids: list[str], fields: dict[str, BugFields], texts: dict[str, str]) -> str:
    def score(i):
        f = fields[i]
        filled = sum(x is not None for x in (f.device, f.os_version, f.game_version, f.platform, f.frequency))
        return filled * 2 + f.has_screenshot + min(len(texts[i]), 120) / 40
    return max(ids, key=score)


def group_rule(msgs: list[Message], fields: dict[str, BugFields]) -> tuple[list[Issue], list[str]]:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vague = [m.id for m in msgs if is_vague(m)]
    real = [m for m in msgs if m.id not in vague]
    texts = {m.id: m.text for m in msgs}
    if not real:
        return [], vague
    docs = [ENV_NOISE.sub(" ", rules.normalize(m.text)) for m in real]
    if len(real) == 1:
        sim = np.ones((1, 1))
    else:
        words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
        chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(docs)
        sim = (cosine_similarity(words) + cosine_similarity(chars)) / 2
    if len(real) == 1:
        labels = [0]
    else:
        labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                         distance_threshold=DISTANCE_THRESHOLD).fit_predict(1 - sim)
    issues = []
    for label in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == label]
        ids = [real[i].id for i in idx]
        if len(idx) > 1:
            sub = sim[np.ix_(idx, idx)]
            conf = float((sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
            conf = round(min(1.0, 0.4 + conf), 2)   # map raw TF-IDF similarity into a readable 0-1 range
        else:
            conf = 1.0
        issues.append(Issue(id="", report_ids=ids, representative_id=representative(ids, fields, texts),
                            confidence=conf))
    return issues, vague


SYSTEM = """You are a live-service QA lead for "Neon Drift", a mobile racing game.
You get bug reports from players (id, extracted summary, original text, possibly Turkish).
Group reports that describe the SAME underlying defect. Same symptom in different words or languages is
the same bug. Different symptoms are different bugs even if the area is related (e.g. a purchase that is
not delivered vs a subscription that is not restored are separate issues).
For each group give: report_ids, representative_id (the clearest, most detailed report),
confidence 0-1 that all reports in the group are the same bug, and a short neutral English title.
Put reports with no concrete symptom ("game broken pls fix") into vague_ids instead of a group.
Every input id must appear exactly once, either in one group or in vague_ids."""


class Group(BaseModel):
    report_ids: list[str]
    representative_id: str
    confidence: float = Field(ge=0, le=1)
    title: str


class Grouping(BaseModel):
    groups: list[Group]
    vague_ids: list[str]


def group_ai(msgs: list[Message], fields: dict[str, BugFields], llm: LLM) -> tuple[list[Issue], list[str]]:
    user = "\n".join(f"[{m.id}] summary: {fields[m.id].summary} | text: {m.text}" for m in msgs)
    result = llm.parse(SMART_MODEL, SYSTEM, user, Grouping)
    valid = {m.id for m in msgs}
    seen: set[str] = set()
    issues = []
    for g in result.groups:
        ids = [i for i in g.report_ids if i in valid and i not in seen]
        if not ids:
            continue
        seen.update(ids)
        rep = g.representative_id if g.representative_id in ids else ids[0]
        issues.append(Issue(id="", report_ids=ids, representative_id=rep, confidence=g.confidence, title=g.title))
    vague = [i for i in result.vague_ids if i in valid and i not in seen]
    seen.update(vague)
    # Never drop a report: anything the model forgot becomes its own issue for a human to look at.
    for m in msgs:
        if m.id not in seen:
            issues.append(Issue(id="", report_ids=[m.id], representative_id=m.id, confidence=0.5))
    return issues, vague


def group(msgs: list[Message], fields: dict[str, BugFields], llm: LLM | None = None):
    return group_ai(msgs, fields, llm) if llm else group_rule(msgs, fields)
