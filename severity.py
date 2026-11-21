"""Stage 4: severity and priority from an explicit rubric.

The rubric itself is plain code so it is deterministic and reviewable. In AI mode Claude only decides
the impact *category* (and explains why); the escalation rules below are always applied the same way.

    Category        Base severity   Meaning
    crash           S1              app closes / cannot start
    data_loss       S1              owned items, progress or currency lost
    payment         S1              real money charged without the item / entitlement
    major_feature   S2              a core feature is broken or blocks progress (soft-lock, freeze, MP)
    minor           S3              feature degraded, workaround exists (audio, timers)
    cosmetic        S4              visual / text only

Escalations
  E1 Money: if >= 1/3 of the reports involve real money, severity is S1.
  E2 Volume: 10+ reports raise severity one level, but volume alone never makes an issue S1.
Priority (P1 = fix now)
  starts equal to severity (S1 -> P1 ... S4 -> P4)
  P-a: 5+ reports -> one level more urgent
  P-b: a single report of an S3/S4 issue -> one level less urgent (wait for confirmation)
  P-c: money involved -> always P1
"""
from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

import rules
from llm import LLM, SMART_MODEL
from models import BugFields, Category, Issue, Message

BASE = {"crash": "S1", "data_loss": "S1", "payment": "S1",
        "major_feature": "S2", "minor": "S3", "cosmetic": "S4"}
ORDER = ["payment", "crash", "data_loss", "major_feature", "minor", "cosmetic"]
VOLUME_THRESHOLD = 10
REACH_THRESHOLD = 5


def _lvl(s: str) -> int:
    return int(s[1])


def category_rule(texts: list[str]) -> tuple[Category, str]:
    frac = {c: sum(rules.has_any(t, rules.CATEGORY_KEYWORDS[c]) > 0 for t in texts) / len(texts) for c in ORDER}
    for c in ORDER:
        if frac[c] >= 0.34:
            return c, f"{frac[c]:.0%} of reports use {c.replace('_', ' ')} keywords"
    best = max(ORDER, key=lambda c: frac[c])
    if frac[best] == 0:
        return "minor", "no clear impact keywords; defaulted to minor"
    return best, f"strongest signal: {best.replace('_', ' ')} ({frac[best]:.0%} of reports)"


def apply_rubric(issue: Issue, category: Category, why: str, money: bool) -> Issue:
    n = len(issue.report_ids)
    notes = [f"Category: {category.replace('_', ' ')} -> base {BASE[category]} ({why})."]
    sev = _lvl(BASE[category])
    if money and sev > 1:
        sev = 1
        notes.append("E1: real money involved -> raised to S1.")
    if n >= VOLUME_THRESHOLD and sev > 2:
        sev -= 1
        notes.append(f"E2: {n} reports -> raised one level.")
    pri = sev
    if money:
        pri = 1
        notes.append("P-c: money involved -> P1.")
    elif n >= REACH_THRESHOLD and pri > 1:
        pri -= 1
        notes.append(f"P-a: {n} reports -> priority raised one level.")
    elif n == 1 and sev >= 3:
        pri = min(4, pri + 1)
        notes.append("P-b: single report of a minor issue -> priority lowered until confirmed.")
    issue.category = category
    issue.severity = f"S{sev}"
    issue.priority = f"P{pri}"
    issue.monetization_impact = money
    issue.severity_reasoning = " ".join(notes)
    return issue


def money_flag(issue: Issue, fields: dict[str, BugFields]) -> bool:
    flagged = sum(fields[i].monetization_impact for i in issue.report_ids)
    return flagged / len(issue.report_ids) >= 1 / 3


SYSTEM = """You are a QA lead assessing player-reported bugs in "Neon Drift", a mobile racing game.
Pick the single impact category that best describes the defect behind these reports:
- crash: app closes or cannot start
- data_loss: owned items, cars, progress or currency lost
- payment: real money charged but the purchase or subscription is not delivered
- major_feature: a core feature is broken or blocks progress (soft-lock, freeze, multiplayer, leaderboard)
- minor: feature degraded, a workaround exists (audio, timers, rewards UI)
- cosmetic: visual or text-only problem
Also say whether real money is involved, and give a one or two sentence reason a developer would accept.
Judge the defect, not how angry the players are."""


class SeverityJudgement(BaseModel):
    category: Category
    money_involved: bool
    reasoning: str


def assess(issues: list[Issue], msgs: dict[str, Message], fields: dict[str, BugFields],
           llm: LLM | None = None) -> list[Issue]:
    for issue in issues:
        texts = [msgs[i].text for i in issue.report_ids]
        if llm:
            user = "\n".join(f"- {t}" for t in texts)
            j = llm.parse(SMART_MODEL, SYSTEM, user, SeverityJudgement, max_tokens=2000)
            apply_rubric(issue, j.category, j.reasoning.rstrip("."), j.money_involved)
        else:
            cat, why = category_rule(texts)
            apply_rubric(issue, cat, why, money_flag(issue, fields))
    return issues


def sort_issues(issues: list[Issue]) -> list[Issue]:
    issues.sort(key=lambda i: (i.priority or "P9", i.severity or "S9", -len(i.report_ids)))
    for n, issue in enumerate(issues, start=1):
        issue.id = f"ND-{n:03d}"
    return issues


def counts(issues: list[Issue]) -> dict:
    return dict(Counter(i.severity for i in issues))
