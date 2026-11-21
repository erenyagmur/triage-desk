"""Runs all stages in order and returns one result dict (the same shape the UI and exports use)."""
from __future__ import annotations

from collections import Counter
from datetime import datetime

import classify
import extract
import group
import severity
import tickets
from llm import LLM
from models import Message


def run(messages: list[Message], llm: LLM | None = None, log=print) -> dict:
    mode = "ai" if llm else "rules"
    by_id = {m.id: m for m in messages}

    cls = classify.classify(messages, llm)
    log(f"[1/6] classified {len(cls)} messages: {dict(Counter(c.type for c in cls))}")
    bug_msgs = [by_id[c.id] for c in cls if c.type == "bug"]

    fields_list = extract.extract(bug_msgs, llm)
    fields = {f.id: f for f in fields_list}
    with_env = sum(1 for f in fields_list if f.device or f.os_version or f.game_version)
    log(f"[2/6] extracted fields from {len(bug_msgs)} bug reports ({with_env} include any environment info)")

    issues, vague = group.group(bug_msgs, fields, llm)
    log(f"[3/6] grouped into {len(issues)} issues, {len(vague)} vague reports set aside")

    severity.assess(issues, by_id, fields, llm)
    severity.sort_issues(issues)
    log(f"[4/6] severity: {dict(sorted(severity.counts(issues).items()))}")

    tickets.draft_tickets(issues, by_id, fields, llm)
    log(f"[5/6] drafted {len(issues)} tickets")

    replies = tickets.draft_replies(bug_msgs, fields, set(vague))
    log(f"[6/6] drafted {len(replies)} replies asking players for missing info")

    cls_by_id = {c.id: c for c in cls}
    return {
        "run_id": datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{mode}",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "messages": {
            m.id: {**m.model_dump(), "type": cls_by_id[m.id].type, "confidence": cls_by_id[m.id].confidence,
                   "fields": fields[m.id].model_dump() if m.id in fields else None}
            for m in messages
        },
        "issues": [i.model_dump() for i in issues],
        "vague_ids": vague,
        "replies": [r.model_dump() for r in replies],
        "stats": {
            "messages": len(messages),
            "by_type": dict(Counter(c.type for c in cls)),
            "bug_reports": len(bug_msgs),
            "issues": len(issues),
            "vague": len(vague),
            "by_severity": dict(sorted(severity.counts(issues).items())),
            "llm_calls": getattr(llm, "calls", 0),
        },
    }
