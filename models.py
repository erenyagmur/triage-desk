"""Pydantic models shared by every pipeline stage (and used as Claude structured-output schemas)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

MessageType = Literal["bug", "feature_request", "feedback", "question", "spam"]
Severity = Literal["S1", "S2", "S3", "S4"]
Priority = Literal["P1", "P2", "P3", "P4"]
Category = Literal["crash", "data_loss", "payment", "major_feature", "minor", "cosmetic"]


class Message(BaseModel):
    id: str
    author: str
    channel: str
    timestamp: str
    text: str


class Classification(BaseModel):
    id: str
    type: MessageType
    confidence: float = Field(ge=0, le=1)


class BugFields(BaseModel):
    id: str
    summary: str = Field(description="One-line neutral summary of the problem, in English")
    platform: Optional[Literal["Android", "iOS"]] = None
    os_version: Optional[str] = None
    device: Optional[str] = None
    game_version: Optional[str] = None
    steps_hint: Optional[str] = Field(None, description="What the player was doing, if stated")
    frequency: Optional[Literal["always", "sometimes", "once"]] = None
    monetization_impact: bool = Field(False, description="Real money spent, lost purchase or subscription")
    has_screenshot: bool = False
    language: Literal["en", "tr"] = "en"

    @property
    def missing(self) -> list[str]:
        out = []
        if not self.device:
            out.append("device")
        if not self.os_version:
            out.append("OS version")
        if not self.game_version:
            out.append("game version")
        if not self.has_screenshot:
            out.append("screenshot")
        return out


class Ticket(BaseModel):
    title: str
    environment: str
    steps_to_reproduce: list[str]
    steps_inferred: bool = Field(description="True if steps were guessed and need verification")
    expected: str
    actual: str
    missing_info: list[str]


class Issue(BaseModel):
    id: str
    report_ids: list[str]
    representative_id: str
    confidence: float = Field(ge=0, le=1, description="How sure we are these reports are one bug")
    title: str = ""
    category: Optional[Category] = None
    severity: Optional[Severity] = None
    priority: Optional[Priority] = None
    severity_reasoning: str = ""
    monetization_impact: bool = False
    ticket: Optional[Ticket] = None


class PlayerReply(BaseModel):
    message_id: str
    author: str
    language: Literal["en", "tr"]
    text: str
