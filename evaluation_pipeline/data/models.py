"""
Pydantic data models for the Dave evaluation pipeline.

Defines the core schemas used across all pipeline stages:
conversation ingestion, evaluation input/output, and scoring.
"""

from __future__ import annotations

import enum
import logging
from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConversationType(str, enum.Enum):
    """Classification tag applied after ingestion."""
    CONTEXT_BACKED = "context_backed"
    CONTEXT_FREE = "context_free"


# ---------------------------------------------------------------------------
# Conversation Record — raw ingestion model
# ---------------------------------------------------------------------------

class ConversationRecord(BaseModel):
    """
    A single Dave conversation as stored / exported from the data source.

    Fields
    ------
    conversation_id : str
        Unique identifier for the conversation.
    user_query : str
        The end-user's question or prompt.
    dave_response : str
        Dave's generated response.
    retrieved_context : Optional[str]
        Supporting documents retrieved for this query, if any.
        ``None`` means the conversation is context-free.
    chat_history : Optional[str]
        Prior turns in the conversation, if available.
    timestamp : datetime
        When the conversation occurred.
    """

    conversation_id: str = Field(
        ..., min_length=1, description="Unique conversation identifier"
    )
    user_query: str = Field(
        ..., min_length=1, description="The user's question or prompt"
    )
    dave_response: str = Field(
        ..., min_length=1, description="Dave's generated response"
    )
    retrieved_context: Optional[str] = Field(
        default=None,
        description="Retrieved supporting documents; None = context-free",
    )
    chat_history: Optional[str] = Field(
        default=None, description="Prior conversation turns"
    )
    expected_intent: Optional[str] = Field(
        default=None,
        description="Ground-truth intent label for testing; None in production",
    )
    retrieved_chunks: Optional[list[str]] = Field(
        default=None,
        description="Individual retrieved chunks if available separately",
    )
    timestamp: datetime = Field(
        ..., description="Conversation timestamp"
    )

    @field_validator("user_query", "dave_response")
    @classmethod
    def must_not_be_blank(cls, v: str, info) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError(
                f"'{info.field_name}' must contain non-whitespace characters"
            )
        return stripped
        
    model_config = {
        "json_schema_extra": {
            "example": {
                "conversation_id": "TEST-001",
                "user_query": "What is HiDevs?",
                "dave_response": "HiDevs is an AI and developer community platform.",
                "retrieved_context": "HiDevs is a developer community platform focused on AI, hackathons, learning and projects.",
                "chat_history": "User: Tell me about HiDevs.",
                "timestamp": "2026-08-07T10:00:00Z",
                "expected_intent": "technical",
                "retrieved_chunks": ["HiDevs is a developer community platform focused on AI, hackathons, learning and projects."]
            }
        }
    }


# ---------------------------------------------------------------------------
# Evaluation Input — cleaned & tagged, ready for evaluators
# ---------------------------------------------------------------------------

class EvaluationInput(BaseModel):
    """
    Cleaned, validated, and tagged conversation ready for evaluator consumption.

    This is the *only* object evaluators should accept.
    """

    conversation_id: str
    user_query: str
    dave_response: str
    retrieved_context: Optional[str] = None
    chat_history: Optional[str] = None
    expected_intent: Optional[str] = None
    retrieved_chunks: Optional[list[str]] = None
    timestamp: datetime
    conversation_type: ConversationType = Field(
        ..., description="Auto-tagged based on retrieved_context presence"
    )


# ---------------------------------------------------------------------------
# Evaluation Result — returned by every evaluator
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """
    Standardised output contract that every evaluator must return.

    Attributes
    ----------
    evaluator_name : str
        Human-readable evaluator identifier (e.g. "response_quality").
    conversation_id : str
        The conversation this result pertains to.
    score : float
        Aggregate score for this evaluation dimension.
    max_score : float
        Maximum achievable score (for normalisation).
    sub_scores : dict[str, float]
        Breakdown of component scores.
    feedback : str
        LLM-generated explanation — must be a *real* generated string,
        never a placeholder or hard-coded value.
    flagged : bool
        Whether this conversation was flagged for human review.
    """

    evaluator_name: str = Field(
        ..., min_length=1, description="Identifier of the evaluator"
    )
    conversation_id: str = Field(
        ..., min_length=1, description="Conversation this result belongs to"
    )
    score: Optional[float] = Field(default=None, ge=0.0, description="Aggregate score")
    max_score: float = Field(..., gt=0.0, description="Maximum possible score")
    applicable: bool = Field(default=True, description="Whether this metric is applicable to the conversation")
    percentage: Optional[float] = Field(
        default=None, description="Optional percentage representation of the score"
    )
    sub_scores: dict[str, Any] = Field(
        default_factory=dict, description="Component score breakdown"
    )
    feedback: str = Field(
        ...,
        min_length=10,
        description="Real, generated explanation — never a placeholder",
    )
    flagged: bool = Field(
        default=False, description="Whether flagged for human review"
    )
    critical_violation: bool = Field(
        default=False, description="Whether a critical safety or policy violation occurred"
    )

    @field_validator("feedback")
    @classmethod
    def feedback_must_be_substantive(cls, v: str) -> str:
        placeholder_markers = {"todo", "placeholder", "not implemented", "n/a"}
        if v.strip().lower() in placeholder_markers:
            raise ValueError(
                "Feedback must be a real, generated explanation — "
                "placeholder strings are not allowed."
            )
        return v
