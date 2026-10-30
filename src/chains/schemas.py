"""Pydantic v2 structured output schema for grounded RAG answers."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Confidence(str, Enum):
    high = "high"
    partial = "partial"
    none = "none"


class SourceCitation(BaseModel):
    """One citation the model claims to have used."""

    label: str = Field(description="Context label such as S1 or [S1]")
    source: str = Field(description="Document file name")
    location: str = Field(default="", description="Page or section pointer")
    snippet: str = Field(default="", description="Optional short supporting quote")


class GroundedResponse(BaseModel):
    """Validated object returned by the RAG chain to all UIs."""

    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: Confidence = Confidence.none
    answered: bool = False
    flags: list[str] = Field(default_factory=list)

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, value: Any) -> Any:
        if value is None:
            return []
        return value

    @classmethod
    def refusal(cls, message: str, *, flags: list[str] | None = None) -> GroundedResponse:
        return cls(
            answer=message,
            sources=[],
            confidence=Confidence.none,
            answered=False,
            flags=flags or ["refusal"],
        )

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
