"""Stage 7: export Jira CSV, a markdown daily triage summary, JSON for the UI, and a data-embedded review page."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).parent
JIRA_PRIORITY = {"P1": "Highest", "P2": "High", "P3": "Medium", "P4": "Low"}
CSV_HEADER = ["Summary", "Issue Type", "Priority", "Severity", "Labels", "Environment", "Description",
              "Linked Reports"]


def jira_description(issue: dict, messages: dict) -> str:
    t = issue["ticket"]
    lines = ["h3. Steps to reproduce"]
    if t["steps_inferred"]:
        lines.append("_Steps marked (inferred, needs verification) were not stated by players._")
    lines += [f"# {s}" for s in t["steps_to_reproduce"]]
    lines += ["h3. Expected", t["expected"], "h3. Actual", t["actual"],
              "h3. Severity reasoning", issue["severity_reasoning"],
              f"h3. Player reports ({len(issue['report_ids'])})"]
    lines += [f"* [{i}] {messages[i]['text']}" for i in issue["report_ids"]]
    lines += ["h3. Missing info QA should collect"] + [f"* {m}" for m in t["missing_info"]]
    lines.append(f"_Drafted by Triage Desk, grouping confidence {issue['confidence']:.2f}. Reviewed by a human before filing._")
    return "\n".join(lines)


def jira_row(issue: dict, messages: dict) -> list[str]:
    labels = ["player-report", "neon-drift", issue["severity"].lower()] + (
        ["monetization"] if issue["monetization_impact"] else [])
    return [issue["ticket"]["title"], "Bug", JIRA_PRIORITY[issue["priority"]], issue["severity"],
            " ".join(labels), issue["ticket"]["environment"], jira_description(issue, messages),
            " ".join(issue["report_ids"])]


def jira_csv(result: dict, issues: list[dict] | None = None) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_HEADER)
    for issue in issues if issues is not None else result["issues"]:
        w.writerow(jira_row(issue, result["messages"]))
    return buf.getvalue()


def daily_summary(result: dict) -> str:
    issues, s = result["issues"], result["stats"]
    s1 = [i for i in issues if i["severity"] == "S1"]
    by_sev = ", ".join(f"{v} {k}" for k, v in s["by_severity"].items())
    out = [f"# Neon Drift daily triage — {result['generated_at'][:10]}", "",
           f"**{s['messages']} messages** → {s['bug_reports']} bug reports → **{s['issues']} issues** "
           f"({by_sev}). Mode: `{result['mode']}`.", "",
           f"**{len(s1)} S1{'s' if len(s1) != 1 else ''}:** " + ("; ".join(
               f"{i['id']} {i['title']} ({len(i['report_ids'])} report{'s' if len(i['report_ids']) != 1 else ''})"
               for i in s1) or "none"), "",
           "| ID | Pri | Sev | Reports | Conf | Title |", "|---|---|---|---|---|---|"]
    out += [f"| {i['id']} | {i['priority']} | {i['severity']} | {len(i['report_ids'])} | {i['confidence']:.2f} | "
            f"{i['title']}{' 💰' if i['monetization_impact'] else ''} |" for i in issues]
    bt = s["by_type"]
    out += ["", "## Not bugs", ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in bt.items() if k != "bug"),
            "", f"## Needs player follow-up",
            f"{len(result['replies'])} reply drafts ready (missing device / OS / version), "
            f"{s['vague']} vague reports with no concrete symptom.", "",
            "_All issues are drafts until approved in the review UI._"]
    return "\n".join(out) + "\n"


def write_all(result: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "triage.json", "csv": out_dir / "jira_import.csv",
             "summary": out_dir / "daily_summary.md", "review": out_dir / "review.html"}
    data = json.dumps(result, indent=2, ensure_ascii=False)
    paths["json"].write_text(data, encoding="utf-8")
    paths["csv"].write_text(jira_csv(result), encoding="utf-8", newline="")
    paths["summary"].write_text(daily_summary(result), encoding="utf-8")
    template = (ROOT / "review.html").read_text(encoding="utf-8")
    embedded = json.dumps(result, ensure_ascii=False).replace("</", "<\\/")
    marker = '<script id="triage-data" type="application/json">null</script>'
    assert marker in template, "review.html is missing the data marker"
    paths["review"].write_text(template.replace(
        marker, f'<script id="triage-data" type="application/json">{embedded}</script>'), encoding="utf-8")
    return paths
