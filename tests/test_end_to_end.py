import csv
import io
import json
from pathlib import Path

import evaluate
import export
import generate_data
import pipeline
from conftest import msg
from llm import FAST_MODEL, SMART_MODEL
from models import Message

quiet = lambda *_: None  # noqa: E731


def _dataset():
    messages, truth, planted = generate_data.generate()
    return [Message(**m) for m in messages], truth, planted


def test_generated_dataset_is_consistent():
    messages, truth, planted = generate_data.generate()
    assert 140 <= len(messages) <= 160 and len(planted) == 12
    assert all(2 <= b["report_count"] <= 15 for b in planted)
    assert {t["id"] for t in truth} == {m["id"] for m in messages}
    assert generate_data.generate() == (messages, truth, planted)   # deterministic


def test_rules_pipeline_end_to_end_and_evaluation():
    msgs, truth, planted = _dataset()
    result = pipeline.run(msgs, None, log=quiet)
    e = evaluate.evaluate(result, truth, planted)
    # Regression floor for rule mode on the synthetic set (not a claim about real data).
    assert e["classification_accuracy"] > 0.9
    assert e["clean_single_issue"] >= 10
    assert e["s1_missed"] == []
    assert all(i["ticket"] for i in result["issues"])


def test_ai_pipeline_with_mocked_api(fake_llm):
    msgs = [msg("m1", "bought coins, charged, nothing"), msg("m2", "para ödedim coin yok")]
    result = pipeline.run(msgs, fake_llm, log=quiet)
    assert result["mode"] == "ai"
    [issue] = result["issues"]
    assert issue["severity"] == "S1" and issue["priority"] == "P1" and issue["monetization_impact"]
    models_used = [m for m, _ in fake_llm.calls]
    assert models_used[:2] == [FAST_MODEL, FAST_MODEL] and set(models_used[2:]) == {SMART_MODEL}


def test_evaluate_perfect_result_scores_full_marks():
    msgs, truth, planted = _dataset()
    gt = {t["id"]: t for t in truth}
    messages = {m.id: {**m.model_dump(), "type": gt[m.id]["type"]} for m in msgs}
    issues = [{"id": b["bug_id"], "severity": b["severity"],
               "report_ids": [t["id"] for t in truth if t["bug_id"] == b["bug_id"]]} for b in planted]
    e = evaluate.evaluate({"mode": "oracle", "messages": messages, "issues": issues, "stats": {}}, truth, planted)
    assert e["classification_accuracy"] == 1 and e["clean_single_issue"] == 12
    assert e["pairwise_f1"] == 1 and e["severity_exact"] == 12 and e["s1_missed"] == []


def test_exports(tmp_path: Path):
    msgs, _, _ = _dataset()
    result = pipeline.run(msgs, None, log=quiet)
    paths = export.write_all(result, tmp_path)
    rows = list(csv.reader(io.StringIO(paths["csv"].read_text(encoding="utf-8"))))
    assert rows[0] == export.CSV_HEADER and len(rows) == len(result["issues"]) + 1
    assert rows[1][2] == "Highest"
    summary = paths["summary"].read_text(encoding="utf-8")
    assert f"**{result['stats']['issues']} issues**" in summary and "S1s:**" in summary
    page = paths["review"].read_text(encoding="utf-8")
    start = page.index('<script id="triage-data" type="application/json">') + len(
        '<script id="triage-data" type="application/json">')
    embedded = json.loads(page[start:page.index("</script>", start)].replace("<\\/", "</"))
    assert embedded["run_id"] == result["run_id"]
