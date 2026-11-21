"""Compare pipeline output with the labeled ground truth, for rule-based and AI mode."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

TYPES = ["bug", "feature_request", "feedback", "question", "spam"]


def _pairs(groups: list[list[str]]) -> set[tuple[str, str]]:
    return {tuple(sorted(p)) for g in groups for p in combinations(g, 2)}


def evaluate(result: dict, truth: list[dict], planted: list[dict]) -> dict:
    gt = {t["id"]: t for t in truth}
    msgs = result["messages"]

    # 1. classification
    correct = sum(msgs[i]["type"] == gt[i]["type"] for i in gt)
    per_type = {}
    for t in TYPES:
        tp = sum(msgs[i]["type"] == t and gt[i]["type"] == t for i in gt)
        pred = sum(msgs[i]["type"] == t for i in gt)
        real = sum(gt[i]["type"] == t for i in gt)
        per_type[t] = {"precision": tp / pred if pred else 0.0, "recall": tp / real if real else 0.0, "n": real}
    confusion = Counter((gt[i]["type"], msgs[i]["type"]) for i in gt if msgs[i]["type"] != gt[i]["type"])
    planted_reports = [i for i in gt if gt[i]["bug_id"]]
    planted_caught = sum(msgs[i]["type"] == "bug" for i in planted_reports)

    # 2. grouping
    issues = result["issues"]
    by_bug = defaultdict(set)
    for i in planted_reports:
        by_bug[gt[i]["bug_id"]].add(i)
    per_bug = []
    for b in planted:
        reports = by_bug[b["bug_id"]]
        holders = [iss for iss in issues if reports & set(iss["report_ids"])]
        best = max(holders, key=lambda iss: len(reports & set(iss["report_ids"])), default=None)
        overlap = len(reports & set(best["report_ids"])) if best else 0
        clean = bool(best) and overlap >= 0.8 * len(reports) and overlap >= 0.8 * len(best["report_ids"])
        sev = best["severity"] if best else None
        per_bug.append({
            "bug_id": b["bug_id"], "title": b["title"], "true_severity": b["severity"], "reports": len(reports),
            "issue": best["id"] if best else None, "issue_size": len(best["report_ids"]) if best else 0,
            "overlap": overlap, "split_into": len(holders), "clean": clean, "pred_severity": sev,
            "s1_found": sev == "S1" and overlap >= 0.5 * len(reports),
        })
    true_pairs = _pairs([list(v) for v in by_bug.values()])
    pred_pairs = _pairs([iss["report_ids"] for iss in issues])
    tp = len(true_pairs & pred_pairs)
    p = tp / len(pred_pairs) if pred_pairs else 0.0
    r = tp / len(true_pairs) if true_pairs else 0.0

    # 3. severity
    sev_exact = sum(b["pred_severity"] == b["true_severity"] for b in per_bug)
    sev_close = sum(b["pred_severity"] is not None and abs(int(b["pred_severity"][1]) - int(b["true_severity"][1])) <= 1
                    for b in per_bug)
    s1 = [b for b in per_bug if b["true_severity"] == "S1"]

    return {
        "mode": result["mode"],
        "llm_calls": result["stats"].get("llm_calls", 0),
        "classification_accuracy": correct / len(gt),
        "per_type": per_type,
        "confusion": {f"{a} -> {b}": n for (a, b), n in confusion.most_common()},
        "planted_reports_classified_bug": planted_caught / len(planted_reports),
        "issues_found": len(issues),
        "planted_bugs": len(planted),
        "clean_single_issue": sum(b["clean"] for b in per_bug),
        "pairwise_precision": p, "pairwise_recall": r, "pairwise_f1": 2 * p * r / (p + r) if p + r else 0.0,
        "severity_exact": sev_exact, "severity_within_one": sev_close,
        "s1_total": len(s1), "s1_missed": [b["bug_id"] for b in s1 if not b["s1_found"]],
        "per_bug": per_bug,
    }


def report(evals: list[dict], skipped: str | None = None) -> str:
    cols = [e["mode"] for e in evals]
    rows = [
        ("Classification accuracy (146 msgs)", lambda e: f"{e['classification_accuracy']:.1%}"),
        ("Planted bug reports classified as bug", lambda e: f"{e['planted_reports_classified_bug']:.1%}"),
        ("Issues produced (12 planted bugs)", lambda e: str(e["issues_found"])),
        ("Planted bugs found as one clean issue", lambda e: f"{e['clean_single_issue']}/{e['planted_bugs']}"),
        ("Duplicate pairs precision / recall", lambda e: f"{e['pairwise_precision']:.2f} / {e['pairwise_recall']:.2f}"),
        ("Duplicate pairs F1", lambda e: f"{e['pairwise_f1']:.2f}"),
        ("Severity exact match", lambda e: f"{e['severity_exact']}/{e['planted_bugs']}"),
        ("Severity within one level", lambda e: f"{e['severity_within_one']}/{e['planted_bugs']}"),
        ("S1 bugs missed", lambda e: f"{len(e['s1_missed'])}/{e['s1_total']}" + (
            f" ({', '.join(e['s1_missed'])})" if e["s1_missed"] else "")),
        ("LLM calls", lambda e: str(e["llm_calls"])),
    ]
    out = ["| Metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    out += [f"| {name} | " + " | ".join(fn(e) for e in evals) + " |" for name, fn in rows]
    if skipped:
        out += ["", f"_{skipped}_"]
    for e in evals:
        out += ["", f"### Per planted bug — {e['mode']} mode", "",
                "| Bug | True sev | Reports | Best issue (size) | Overlap | Split into | Clean | Pred sev |",
                "|---|---|---|---|---|---|---|---|"]
        out += [f"| {b['bug_id']} {b['title']} | {b['true_severity']} | {b['reports']} | "
                f"{b['issue'] or '—'} ({b['issue_size']}) | {b['overlap']} | {b['split_into']} | "
                f"{'✅' if b['clean'] else '❌'} | {b['pred_severity'] or '—'} |" for b in e["per_bug"]]
        if e["confusion"]:
            out += ["", "Misclassifications (truth -> predicted): " +
                    ", ".join(f"{k}: {v}" for k, v in e["confusion"].items())]
    return "\n".join(out) + "\n"
