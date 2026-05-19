"""Pydantic schemas for tool outputs and final answers."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Uniform shape returned by all tools."""

    ok: bool
    data: Any = None
    error: str | None = None


class Candidate(BaseModel):
    name: str
    lat: float | None = None
    lng: float | None = None
    address: str | None = None
    clues_matched: list[str] = Field(default_factory=list)
    notes: str = ""


class FinalAnswer(BaseModel):
    location_name: str
    address: str | None = None
    lat: float | None = None
    lng: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    clue_mapping: dict[str, str] = Field(default_factory=dict)
    candidates_considered: list[Candidate] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
