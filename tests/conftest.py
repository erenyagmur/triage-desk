import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import BugFields, Classification, Message, Ticket  # noqa: E402


def msg(i, text, author="racer_1000"):
    return Message(id=i, author=author, channel="#bug-reports", timestamp="2026-11-20T10:00", text=text)


class FakeLLM:
    """Stands in for llm.LLM. Returns canned structured outputs per schema; never touches the network."""

    def __init__(self, overrides=None):
        self.calls = []
        self.overrides = overrides or {}

    def parse(self, model, system, user, schema, max_tokens=16000):
        self.calls.append((model, schema.__name__))
        if schema.__name__ in self.overrides:
            return self.overrides[schema.__name__](user)
        ids = re.findall(r"\[(m\d+)\]", user)
        name = schema.__name__
        if name == "ClassificationBatch":
            return schema(items=[Classification(id=i, type="bug", confidence=0.9) for i in ids])
        if name == "ExtractionBatch":
            return schema(items=[BugFields(id=i, summary="Coins not delivered after purchase", platform="Android",
                                           monetization_impact=True) for i in ids])
        if name == "Grouping":
            return schema(groups=[{"report_ids": ids, "representative_id": ids[0], "confidence": 0.9,
                                   "title": "Shop: coins not delivered"}], vague_ids=[])
        if name == "SeverityJudgement":
            return schema(category="payment", money_involved=True, reasoning="Players charged without coins.")
        if name == "Ticket":
            return Ticket(title="Shop: coins not delivered after purchase", environment="Android",
                          steps_to_reproduce=["Buy coin pack (inferred, needs verification)"], steps_inferred=True,
                          expected="Coins granted", actual="No coins", missing_info=["order IDs"])
        raise AssertionError(f"unexpected schema {name}")


@pytest.fixture
def fake_llm():
    return FakeLLM()
