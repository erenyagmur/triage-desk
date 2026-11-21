"""Stages 5 + 6: draft a Jira-ready ticket per issue, and friendly replies asking players for missing info."""
from __future__ import annotations

from collections import Counter

from llm import LLM, SMART_MODEL
from models import BugFields, Issue, Message, PlayerReply, Ticket

INFERRED = " (inferred, needs verification)"

STEPS = {
    "crash": ["Install Neon Drift {ver} on a device running {os}", "Launch the game from the home screen",
              "Watch the splash / loading screen"],
    "payment": ["Log in with a test account and note the coin balance / VIP status",
                "Buy the item from the reports using a sandbox payment method",
                "Complete the store purchase flow",
                "Check the coin balance or entitlement after the store confirms payment"],
    "data_loss": ["Use an account that owns several upgraded cars on the previous version",
                  "Update the game to {ver}", "Open the Garage", "Check that every owned car and upgrade is listed"],
}
EXPECTED = {
    "crash": "The game reaches the main menu.",
    "payment": "Purchased coins / entitlement are granted immediately after the store confirms payment.",
    "data_loss": "All owned cars, upgrades and progress are kept after the update.",
    "major_feature": "The feature works and the player can continue.",
    "minor": "The feature behaves normally without restarting the game.",
    "cosmetic": "UI and visuals render correctly.",
}


def environment(issue: Issue, fields: dict[str, BugFields]) -> tuple[str, list[str]]:
    fs = [fields[i] for i in issue.report_ids]
    parts, missing = [], []
    for label, key in [("Platform", "platform"), ("OS", "os_version"), ("Device", "device"),
                       ("Game version", "game_version")]:
        vals = Counter(getattr(f, key) for f in fs if getattr(f, key))
        if vals:
            parts.append(f"{label}: " + ", ".join(f"{v} ({c})" for v, c in vals.most_common(4)))
        if sum(vals.values()) < len(fs) / 2:
            missing.append(label.lower())
    reported = sum(1 for f in fs if any((f.device, f.os_version, f.game_version)))
    env = "; ".join(parts) if parts else "Not provided by players"
    return f"{env}. Environment given in {reported}/{len(fs)} reports.", missing


AREAS = {"zvip": "VIP subscription", "zcoins": "Shop / IAP", "zcrash": "Launch", "ztutorial": "Tutorial",
         "zleaderboard": "Leaderboard", "zonline": "Multiplayer", "zfreeze": "Race", "zaudio": "Audio",
         "zdaily": "Daily reward", "zpaint": "Garage visuals", "zlocale": "Localization (TR)",
         "zgarage_car": "Garage"}


def area(issue: Issue, msgs: dict[str, Message]) -> str:
    """Most common game area among the reports, in a fixed preference order."""
    import rules
    c = Counter(x for i in issue.report_ids for x in rules.concepts(msgs[i].text))
    ranked = [a for a in AREAS if c[a] >= len(issue.report_ids) / 2] or [a for a, _ in c.most_common(1)]
    return AREAS.get(ranked[0], "Game") if ranked else "Game"


def ticket_rule(issue: Issue, msgs: dict[str, Message], fields: dict[str, BugFields]) -> Ticket:
    rep = fields[issue.representative_id]
    env, missing = environment(issue, fields)
    vers = Counter(fields[i].game_version for i in issue.report_ids if fields[i].game_version)
    oses = Counter(fields[i].os_version for i in issue.report_ids if fields[i].os_version)
    ver = vers.most_common(1)[0][0] if vers else "(latest build)"
    os_ = oses.most_common(1)[0][0] if oses else "(OS from reports)"
    tmpl = STEPS.get(issue.category or "", [
        f'Follow the scenario in the representative report: "{msgs[issue.representative_id].text}"',
        "Observe the result"])
    steps = [s.format(ver=ver, os=os_) + INFERRED for s in tmpl]
    missing_info = [f"{m} (missing in most reports)" for m in missing] + ["confirmed reproduction steps"]
    if issue.category == "payment":
        missing_info.append("store order / transaction IDs for affected purchases")
    if not any(fields[i].has_screenshot for i in issue.report_ids):
        missing_info.append("screenshot or screen recording")
    title = issue.title or f"{area(issue, msgs)}: {rep.summary}"
    return Ticket(title=title[:100], environment=env, steps_to_reproduce=steps, steps_inferred=True,
                  expected=EXPECTED.get(issue.category or "", "Works as designed."),
                  actual=rep.summary, missing_info=missing_info)


SYSTEM = """You are a QA lead writing a Jira bug ticket for "Neon Drift", a mobile racing game, from player
reports that were already grouped as one issue. Rules:
- Title: short, specific, in the form "<Area>: <symptom> [<condition>]".
- Environment: only devices / OS / versions the players actually gave; say how many reports had it.
- Steps to reproduce: concrete steps a tester can follow. If any step is a guess rather than something
  a player stated, set steps_inferred=true and append " (inferred, needs verification)" to that step.
- Expected vs actual: one sentence each.
- missing_info: what QA still has to collect from players or logs before a developer can act.
Do not invent facts. Write in English even if reports are Turkish."""


def ticket_ai(issue: Issue, msgs: dict[str, Message], fields: dict[str, BugFields], llm: LLM) -> Ticket:
    env, _ = environment(issue, fields)
    reports = "\n".join(f"- [{i}] {msgs[i].text}" for i in issue.report_ids)
    user = (f"Issue severity {issue.severity}, priority {issue.priority}, category {issue.category}.\n"
            f"Aggregated environment: {env}\nRepresentative report: {msgs[issue.representative_id].text}\n"
            f"All reports:\n{reports}")
    return llm.parse(SMART_MODEL, SYSTEM, user, Ticket, max_tokens=4000)


def draft_tickets(issues, msgs, fields, llm: LLM | None = None) -> list[Issue]:
    for issue in issues:
        issue.ticket = ticket_ai(issue, msgs, fields, llm) if llm else ticket_rule(issue, msgs, fields)
        if not issue.title:
            issue.title = issue.ticket.title
    return issues


ASK_EN = {"device": "your phone model", "OS version": "Android/iOS version",
          "game version": "game version (Settings > About)", "screenshot": "a screenshot or short video"}
ASK_TR = {"device": "telefon modelin", "OS version": "Android/iOS sürümün",
          "game version": "oyun sürümü (Ayarlar > Hakkında)", "screenshot": "bir ekran görüntüsü veya kısa video"}


def draft_replies(bug_msgs: list[Message], fields: dict[str, BugFields], vague_ids: set[str]) -> list[PlayerReply]:
    """Template replies (not AI) so tone and promises stay consistent; a human sends them, never the tool."""
    replies = []
    for m in bug_msgs:
        f = fields[m.id]
        missing = f.missing
        env_missing = [x for x in missing if x != "screenshot"]
        if m.id not in vague_ids and len(env_missing) < 2:
            continue
        if f.language == "tr":
            asks = ", ".join(ASK_TR[x] for x in missing)
            lead = "Ne olduğunu biraz daha anlatır mısın? Hangi ekrandaydın, ne yapıyordun? " if m.id in vague_ids else ""
            text = (f"Selam @{m.author}, haber verdiğin için teşekkürler! {lead}"
                    f"İnceleyebilmemiz için şunları paylaşabilir misin: {asks}. Hemen bakacağız 🙏")
        else:
            asks = ", ".join(ASK_EN[x] for x in missing)
            lead = "Could you tell us a bit more about what happened and where in the game? " if m.id in vague_ids else ""
            text = (f"Hey @{m.author}, thanks for reporting this! {lead}"
                    f"To help us track it down, could you share: {asks}? We're on it 🙏")
        replies.append(PlayerReply(message_id=m.id, author=m.author, language=f.language, text=text))
    return replies
