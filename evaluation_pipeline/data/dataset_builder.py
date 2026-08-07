"""
Dataset builder — transforms raw ConversationRecords into validated,
tagged EvaluationInput objects ready for evaluator consumption.

Responsibilities:
  • Load conversation records from any source (mock or production).
  • Auto-tag each record as context_backed / context_free.
  • Normalize and validate all fields.
  • Surface clear errors for malformed records without crashing the pipeline.
"""

from __future__ import annotations

import logging
from typing import Sequence

from pydantic import ValidationError

from evaluation_pipeline.data.models import (
    ConversationRecord,
    ConversationType,
    EvaluationInput,
)

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds a clean, validated evaluation dataset from raw conversation records.

    Usage
    -----
    >>> from evaluation_pipeline.data.mock_conversations import get_mock_conversations
    >>> builder = DatasetBuilder()
    >>> inputs = builder.build(get_mock_conversations())
    >>> print(len(inputs))
    15
    """

    def build(
        self,
        records: Sequence[ConversationRecord],
    ) -> list[EvaluationInput]:
        """
        Transform raw conversation records into evaluation-ready inputs.

        Parameters
        ----------
        records : Sequence[ConversationRecord]
            Raw conversation records to process.

        Returns
        -------
        list[EvaluationInput]
            Validated, tagged inputs ready for evaluators.
            Records that fail validation are logged and skipped.
        """
        logger.info("Building evaluation dataset from %d record(s).", len(records))

        evaluation_inputs: list[EvaluationInput] = []
        skipped = 0

        for record in records:
            try:
                evaluation_input = self._process_record(record)
                evaluation_inputs.append(evaluation_input)
            except (ValidationError, ValueError) as exc:
                skipped += 1
                logger.warning(
                    "Skipped record '%s' due to validation error: %s",
                    getattr(record, "conversation_id", "<unknown>"),
                    exc,
                )

        logger.info(
            "Dataset build complete: %d valid input(s), %d skipped.",
            len(evaluation_inputs),
            skipped,
        )

        if not evaluation_inputs:
            logger.error(
                "No valid evaluation inputs produced! "
                "Check source data for systemic issues."
            )

        return evaluation_inputs

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _process_record(self, record: ConversationRecord) -> EvaluationInput:
        """Validate, normalize, and tag a single conversation record."""

        # Normalize text fields
        user_query = self._normalize_text(record.user_query)
        dave_response = self._normalize_text(record.dave_response)
        retrieved_context = (
            self._normalize_text(record.retrieved_context)
            if record.retrieved_context
            else None
        )
        chat_history = (
            self._normalize_text(record.chat_history)
            if record.chat_history
            else None
        )

        # Auto-tag conversation type
        conversation_type = self._classify(retrieved_context)

        logger.debug(
            "Processed record '%s' → type=%s",
            record.conversation_id,
            conversation_type.value,
        )

        return EvaluationInput(
            conversation_id=record.conversation_id,
            user_query=user_query,
            dave_response=dave_response,
            retrieved_context=retrieved_context,
            chat_history=chat_history,
            timestamp=record.timestamp,
            conversation_type=conversation_type,
        )

    @staticmethod
    def _classify(retrieved_context: str | None) -> ConversationType:
        """
        Classify a conversation based on the presence of retrieved context.

        A non-None, non-empty-after-stripping context means context_backed.
        """
        if retrieved_context and retrieved_context.strip():
            return ConversationType.CONTEXT_BACKED
        return ConversationType.CONTEXT_FREE

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Apply basic text normalization:
          • Strip leading/trailing whitespace.
          • Collapse runs of 3+ newlines into double newlines.
          • Ensure no trailing whitespace on individual lines.
        """
        import re

        text = text.strip()
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing whitespace on each line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text
