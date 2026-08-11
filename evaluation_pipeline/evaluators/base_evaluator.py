"""
Abstract base evaluator interface.

All concrete evaluators (response_quality, groundedness, safety, intent_understanding, memory_and_continuity) MUST
inherit from BaseEvaluator and implement the `evaluate` method, returning
an EvaluationResult with real, LLM-generated feedback — never placeholders.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from evaluation_pipeline.data.models import EvaluationInput, EvaluationResult

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluation dimensions.

    Subclasses must:
      1. Set ``self.name`` to a human-readable evaluator identifier.
      2. Implement ``evaluate()`` returning an ``EvaluationResult``.
      3. Ensure ``feedback`` in the result is a *real*, generated explanation.

    Example
    -------
    >>> class MyEvaluator(BaseEvaluator):
    ...     name = "my_evaluator"
    ...     def evaluate(self, input: EvaluationInput) -> EvaluationResult:
    ...         # ... real evaluation logic ...
    ...         return EvaluationResult(...)
    """

    name: str = "base_evaluator"

    @abstractmethod
    def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
        """
        Run evaluation on a single conversation.

        Parameters
        ----------
        eval_input : EvaluationInput
            A validated, tagged conversation ready for evaluation.

        Returns
        -------
        EvaluationResult
            Evaluation scores, feedback, and flag status.
            The ``feedback`` field MUST contain a real, generated explanation.
        """
        ...

    def evaluate_batch(
        self, inputs: list[EvaluationInput]
    ) -> list[EvaluationResult]:
        """
        Evaluate a batch of conversations sequentially.

        Override in subclasses for concurrent/parallel evaluation.

        Parameters
        ----------
        inputs : list[EvaluationInput]
            Batch of validated conversations.

        Returns
        -------
        list[EvaluationResult]
            One result per input. Failed evaluations are logged and
            returned with flagged=True and error details in feedback.
        """
        results: list[EvaluationResult] = []

        for eval_input in inputs:
            try:
                result = self.evaluate(eval_input)
                results.append(result)
                score_str = f"{result.score:.2f}" if result.score is not None else "N/A"
                logger.info(
                    "[%s] Evaluated '%s': score=%s/%s, flagged=%s",
                    self.name,
                    eval_input.conversation_id,
                    score_str,
                    f"{result.max_score:.2f}",
                    result.flagged,
                )
            except Exception as exc:
                logger.error(
                    "[%s] Failed to evaluate '%s': %s",
                    self.name,
                    eval_input.conversation_id,
                    exc,
                    exc_info=True,
                )
                # Return a flagged error result rather than crashing the batch
                error_result = EvaluationResult(
                    evaluator_name=self.name,
                    conversation_id=eval_input.conversation_id,
                    score=None,
                    max_score=20.0,
                    status="failed",
                    sub_scores={},
                    feedback=f"Evaluation failed with error: {exc}",
                    flagged=True,
                )
                results.append(error_result)

        return results
