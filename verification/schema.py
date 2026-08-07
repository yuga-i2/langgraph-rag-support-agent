"""
Pydantic mirror of data/output_schema.json.

Using a real model (instead of hand-rolled dict checks) means "follows the
required output schema" in the Verification node is an actual schema
validation call, not a vibe check.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Classification(str, Enum):
    ANSWERABLE = "answerable"
    REQUIRES_CLARIFICATION = "requires_clarification"
    REQUIRES_ESCALATION = "requires_escalation"
    OUT_OF_SCOPE = "out_of_scope"
    SAFE_FAILURE = "safe_failure"


class SourceRef(BaseModel):
    source_id: str = Field(..., min_length=1, description="KB document ID or resolved-case ID")
    passage: str = Field(..., min_length=1, description="Relevant excerpt or passage identifier")


class SupportResponse(BaseModel):
    classification: Classification
    answer: str = Field(..., min_length=1)
    sources: List[SourceRef] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_human: bool
    reason: str = Field(..., min_length=1)
    clarification_question: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @field_validator("sources")
    @classmethod
    def _answerable_needs_sources(cls, v, info):
        classification = info.data.get("classification")
        if classification == Classification.ANSWERABLE and len(v) == 0:
            raise ValueError("answerable responses must cite at least one source")
        return v

    model_config = {"use_enum_values": True}
