from types import SimpleNamespace

import pytest

import classify
import extract
import group
import severity
import tickets
from conftest import msg
from llm import FAST_MODEL, LLM, SMART_MODEL, LLMError
from models import BugFields, Issue


# ---------- classify ----------
@pytest.mark.parametrize("text,expected", [
    ("game crashes on startup since android 14", "bug"),
    ("satın aldım para çekildi ama coinler gelmedi", "bug"),
    ("how do i unlock the night city track?", "question"),
    ("please add a photo mode", "feature_request"),
    ("FREE COINS generator click here", "spam"),
    ("🔥🔥🔥", "spam"),
    ("too many ads honestly", "feedback"),
])
def test_classify_rule(text, expected):
    assert classify.classify_rule(msg("m1", text)).type == expected


def test_classify_ai_uses_haiku_and_falls_back_for_skipped_ids(fake_llm):
    from classify import ClassificationBatch
    fake_llm.overrides["ClassificationBatch"] = lambda user: ClassificationBatch(items=[])  # model skipped all
    out = classify.classify([msg("m1", "please add a photo mode")], fake_llm)
    assert out[0].type == "feature_request"          # rule fallback, nothing dropped
    assert fake_llm.calls[0][0] == FAST_MODEL


# ---------- extract ----------
def test_extract_rule_fields():
    f = extract.extract_rule(msg("m1", "Launch crash - Galaxy A54, Android 14, v2.4.1. ss attached"))
    assert f.platform == "Android" and f.os_version == "Android 14" and f.game_version == "2.4.1"
    assert f.device and "A54" in f.device and f.has_screenshot
    assert f.missing == []


def test_extract_rule_ios_and_money_and_turkish():
    f = extract.extract_rule(msg("m1", "para ödedim coin yok!! iOS 17.4"))
    assert f.os_version == "iOS 17.4" and f.platform == "iOS"
    assert f.monetization_impact and f.language == "tr"
    assert "device" in f.missing


# ---------- group ----------
def _fields(ms):
    return {m.id: extract.extract_rule(m) for m in ms}


def test_group_rule_merges_duplicates_and_sets_aside_vague():
    ms = [msg("m1", "game crashes on startup android 14"), msg("m2", "crash at the loading logo every time"),
          msg("m3", "oyun açılmıyor, logoda kapanıyor"), msg("m4", "no sound after watching an ad"),
          msg("m5", "audio cuts out after ad until restart"), msg("m6", "game broken pls fix")]
    issues, vague = group.group_rule(ms, _fields(ms))
    sets = [set(i.report_ids) for i in issues]
    assert {"m1", "m2", "m3"} in sets and {"m4", "m5"} in sets
    assert vague == ["m6"]
    assert all(0 <= i.confidence <= 1 for i in issues)


def test_group_ai_repairs_bad_model_output(fake_llm):
    from group import Grouping
    fake_llm.overrides["Grouping"] = lambda user: Grouping(groups=[
        {"report_ids": ["m1", "m2", "m99"], "representative_id": "m99", "confidence": 0.8, "title": "x"},
        {"report_ids": ["m2"], "representative_id": "m2", "confidence": 0.5, "title": "dup"},
    ], vague_ids=[])
    ms = [msg("m1", "crash"), msg("m2", "crash on start"), msg("m3", "leaderboard stuck")]
    issues, vague = group.group_ai(ms, _fields(ms), fake_llm)
    assert [i.report_ids for i in issues] == [["m1", "m2"], ["m3"]]   # unknown id dropped, dup removed, m3 kept
    assert issues[0].representative_id == "m1"
    assert fake_llm.calls[0][0] == SMART_MODEL


# ---------- severity rubric ----------
def _issue(n):
    return Issue(id="", report_ids=[f"m{i}" for i in range(n)], representative_id="m0", confidence=1)


def test_rubric_money_forces_s1_p1():
    i = severity.apply_rubric(_issue(2), "minor", "x", money=True)
    assert (i.severity, i.priority) == ("S1", "P1")


def test_rubric_volume_raises_but_never_to_s1():
    assert severity.apply_rubric(_issue(10), "minor", "x", False).severity == "S2"
    assert severity.apply_rubric(_issue(15), "major_feature", "x", False).severity == "S2"


def test_rubric_priority_reach_and_single_report():
    assert severity.apply_rubric(_issue(6), "major_feature", "x", False).priority == "P1"
    single = severity.apply_rubric(_issue(1), "cosmetic", "x", False)
    assert (single.severity, single.priority) == ("S4", "P4")
    assert severity.apply_rubric(_issue(1), "minor", "x", False).priority == "P4"


def test_sort_issues_by_priority_and_assigns_ids():
    a = severity.apply_rubric(_issue(1), "cosmetic", "x", False)
    b = severity.apply_rubric(_issue(3), "crash", "x", False)
    out = severity.sort_issues([a, b])
    assert [i.severity for i in out] == ["S1", "S4"] and out[0].id == "ND-001"


# ---------- tickets + replies ----------
def test_rule_ticket_marks_inferred_steps_and_missing_info():
    ms = {m.id: m for m in [msg("m1", "bought coins, charged, no coins"), msg("m2", "paid 4.99 nothing came")]}
    fields = {k: extract.extract_rule(v) for k, v in ms.items()}
    issue = severity.apply_rubric(Issue(id="ND-001", report_ids=["m1", "m2"], representative_id="m1",
                                        confidence=0.8), "payment", "x", True)
    t = tickets.ticket_rule(issue, ms, fields)
    assert t.steps_inferred and all(s.endswith("(inferred, needs verification)") for s in t.steps_to_reproduce)
    assert any("transaction" in m for m in t.missing_info)
    assert t.title.startswith("Shop / IAP:")


def test_replies_only_when_info_missing_and_in_player_language():
    ms = [msg("m1", "oyun açılmıyor", author="racer_1"),
          msg("m2", "Launch crash - Galaxy A54, Android 14, v2.4.1", author="racer_2")]
    replies = tickets.draft_replies(ms, _fields(ms), vague_ids=set())
    assert [r.message_id for r in replies] == ["m1"]
    assert replies[0].language == "tr" and "@racer_1" in replies[0].text


# ---------- LLM wrapper against a fake Anthropic client ----------
class FakeClient:
    def __init__(self, stop_reason="end_turn", parsed=None):
        self.seen = []
        resp = SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed)

        def parse(**kw):
            self.seen.append(kw)
            return resp
        self.messages = SimpleNamespace(parse=parse)
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=parse))


def test_llm_wrapper_routes_models_and_uses_fallbacks_for_sonnet():
    parsed = BugFields(id="m1", summary="x")
    c = FakeClient(parsed=parsed)
    llm = LLM(client=c, cache=False)
    assert llm.parse(FAST_MODEL, "s", "u", BugFields) is parsed
    llm.parse(SMART_MODEL, "s", "u", BugFields)
    assert "fallbacks" not in c.seen[0] and c.seen[0]["output_format"] is BugFields
    assert c.seen[1]["fallbacks"] == "default" and c.seen[1]["model"] == SMART_MODEL
    assert llm.calls == 2


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_llm_wrapper_raises_on_refusal_and_truncation(stop):
    with pytest.raises(LLMError):
        LLM(client=FakeClient(stop_reason=stop), cache=False).parse(FAST_MODEL, "s", "u", BugFields)
