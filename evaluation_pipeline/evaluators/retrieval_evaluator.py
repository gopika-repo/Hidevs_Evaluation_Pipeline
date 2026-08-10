"""
Retrieval Quality Evaluator — Phase 1C

Evaluates RAG retrieval quality for context_backed conversations.
For context_free conversations, returns non-applicable.

Scoring (Max 15):
  - Context Precision (Max 6): (Ragas context_precision) * 6
  - Context Recall (Max 5): (Ragas context_recall) * 5
  - Noise / Redundancy (Max 4): (1 - (Irrelevant or Duplicate Chunks / Total Chunks)) * 4 (from LLM judge)
"""

from __future__ import annotations

import logging
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Required compatibility patch — see ragas_compat_patch.py for details
import evaluation_pipeline.utils.ragas_compat_patch


from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from datasets import Dataset

from evaluation_pipeline.data.models import (
    ConversationType,
    EvaluationInput,
    EvaluationResult,
)
from evaluation_pipeline.evaluators.base_evaluator import BaseEvaluator
from evaluation_pipeline.utils.llm_client import LLMJudge
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Constants
PRECISION_WEIGHT = 6.0
RECALL_WEIGHT = 5.0
NOISE_WEIGHT = 4.0
MAX_SCORE = 15.0

_SYSTEM_PROMPT = """\
You are a STRICT, expert evaluation judge assessing RAG retrieval quality.
You must be critical and rigorous.

Your task is to:
Given the user query, retrieved chunks, and retrieved context:
1. Assess the coverage/recall of the retrieved chunks. Provide a coverage score (1-5):
   - 5 = All necessary information to answer the user query is covered by the chunks.
   - 3 = Core information is covered, but minor details are missing.
   - 1 = No relevant information is covered.
2. Identify duplicate or completely irrelevant chunks (noise). Count them.

Return your evaluation as a JSON object with EXACTLY this structure (no extra keys):
{
  "relevant_chunk_count": <int>,
  "total_chunk_count": <int>,
  "coverage_score": {"score": <1-5>, "reasoning": "<why this coverage score was assigned>"},
  "duplicate_or_irrelevant_count": <int>,
  "explanation": "<overall explanation of coverage and noise>"
}
"""

def _build_user_prompt(eval_input: EvaluationInput) -> str:
    """Construct the user prompt for noise and coverage assessment."""
    chunks_str = ""
    if eval_input.retrieved_chunks:
        for idx, chunk in enumerate(eval_input.retrieved_chunks):
            chunks_str += f"### Chunk {idx + 1}\n{chunk}\n\n"
    else:
        chunks_str = eval_input.retrieved_context or ""

    parts = [
        "Assess the coverage and noise/redundancy of the following retrieved chunks.\n",
        f"## User Query\n{eval_input.user_query}\n",
        f"## Retrieved Chunks\n{chunks_str}",
    ]

    parts.append(
        "\nProvide the count of relevant and total chunks, the coverage score (1-5), and duplicate/irrelevant chunk counts. "
        "Return ONLY valid JSON."
    )
    return "\n".join(parts)


class RetrievalEvaluator(BaseEvaluator):
    """
    Evaluator for Retrieval Quality.
    Uses Ragas for precision/recall and LLM judge for Noise/Redundancy.
    Max Score: 15.0.
    """

    name: str = "retrieval_quality"

    def __init__(self) -> None:
        self._judge = LLMJudge()
        # Initialize Gemini for Ragas
        self.ragas_llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        context_precision.llm = self.ragas_llm
        context_recall.llm = self.ragas_llm
        logger.info("RetrievalEvaluator initialized.")

    def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult:
        """Run retrieval evaluation or return not applicable."""
        if eval_input.conversation_type != ConversationType.CONTEXT_BACKED:
            return EvaluationResult(
                evaluator_name=self.name,
                conversation_id=eval_input.conversation_id,
                score=None,
                max_score=MAX_SCORE,
                applicable=False,
                feedback="Not applicable — no retrieved context in this conversation.",
                flagged=False,
            )

        logger.debug(
            "Evaluating retrieval quality for '%s'", eval_input.conversation_id
        )

        user_query = eval_input.user_query
        dave_response = eval_input.dave_response
        retrieved_chunks = eval_input.retrieved_chunks or [eval_input.retrieved_context or ""]
        retrieved_context = eval_input.retrieved_context or ""

        # Run Ragas and LLM judge concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_ragas = executor.submit(
                self._run_ragas_evaluation, user_query, dave_response, retrieved_chunks, retrieved_context
            )
            future_llm = executor.submit(
                self._run_llm_judge, eval_input
            )

            # Collect results
            ragas_result = future_ragas.result()
            llm_result = future_llm.result()

        # Parse Ragas metrics defensively
        ragas_precision = 0.0
        ragas_recall = 0.0
        if isinstance(ragas_result, dict):
            ragas_precision = float(ragas_result.get("context_precision", 0.0) or 0.0)
            ragas_recall = float(ragas_result.get("context_recall", 0.0) or 0.0)
        else:
            try:
                d = dict(ragas_result)
                ragas_precision = float(d.get("context_precision", 0.0) or 0.0)
                ragas_recall = float(d.get("context_recall", 0.0) or 0.0)
            except Exception:
                try:
                    ragas_precision = float(getattr(ragas_result, "context_precision", 0.0) or 0.0)
                    ragas_recall = float(getattr(ragas_result, "context_recall", 0.0) or 0.0)
                except Exception:
                    pass

        import math
        def clean_val(v) -> float:
            try:
                val = float(v)
                if math.isnan(val) or math.isinf(val):
                    return 0.0
                return val
            except (TypeError, ValueError):
                return 0.0

        ragas_precision = clean_val(ragas_precision)
        ragas_recall = clean_val(ragas_recall)

        # Parse LLM results
        raw_coverage = self._extract_score(llm_result, "coverage_score")
        total_chunks = int(llm_result.get("total_chunk_count", len(retrieved_chunks)) or len(retrieved_chunks))
        if total_chunks <= 0:
            total_chunks = len(retrieved_chunks) or 1
        dup_or_irrelevant = int(llm_result.get("duplicate_or_irrelevant_count", 0) or 0)

        # Compute sub-scores
        precision_score = round(ragas_precision * PRECISION_WEIGHT, 2)
        recall_score = round(ragas_recall * RECALL_WEIGHT, 2)
        
        # Noise score: (1 - (dup_or_irrelevant / total)) * 4.0
        noise_ratio = min(1.0, max(0.0, dup_or_irrelevant / total_chunks))
        noise_score = round((1.0 - noise_ratio) * NOISE_WEIGHT, 2)

        # LLM coverage cross-check
        llm_coverage_score = round((raw_coverage / 5.0) * RECALL_WEIGHT, 2)

        sub_scores = {
            "context_precision": precision_score,
            "context_recall": recall_score,
            "noise_redundancy": noise_score,
            "llm_coverage_check": llm_coverage_score,
        }

        # If there's a major discrepancy between Ragas recall and LLM recall check, log it
        final_recall_score = recall_score
        if abs(recall_score - llm_coverage_score) > 1.5:
            logger.warning(
                "Recall discrepancy for %s: Ragas=%s, LLM=%s. Overriding with LLM check.",
                eval_input.conversation_id,
                recall_score,
                llm_coverage_score,
            )
            final_recall_score = llm_coverage_score

        total_score = round(precision_score + final_recall_score + noise_score, 2)

        # Build feedback
        feedback_parts = [
            f"Context Precision (Ragas): {precision_score:.2f}/{PRECISION_WEIGHT:.0f} (raw: {ragas_precision:.4f})",
            f"Context Recall (Ragas): {recall_score:.2f}/{RECALL_WEIGHT:.0f} (raw: {ragas_recall:.4f})",
            f"LLM Coverage Check: {llm_coverage_score:.2f}/{RECALL_WEIGHT:.0f} (raw score: {raw_coverage}/5)",
            f"Noise / Redundancy: {noise_score:.2f}/{NOISE_WEIGHT:.0f} (identified {dup_or_irrelevant} duplicate/irrelevant chunk(s) out of {total_chunks})",
            f"LLM Reasoning: {llm_result.get('coverage_score', {}).get('reasoning', '')}",
            f"Explanation: {llm_result.get('explanation', '')}"
        ]
        feedback = "\n\n".join(feedback_parts)

        # Flag if score is below 50%
        flagged = total_score < (MAX_SCORE * 0.5)

        return EvaluationResult(
            evaluator_name=self.name,
            conversation_id=eval_input.conversation_id,
            score=total_score,
            max_score=MAX_SCORE,
            sub_scores=sub_scores,
            feedback=feedback,
            flagged=flagged,
            applicable=True,
        )

    def _run_ragas_evaluation(
        self, question: str, answer: str, contexts: list[str], ground_truth: str
    ) -> dict[str, Any]:
        """Call Ragas evaluate with Gemini."""
        data = {
            "question": [question],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
            "answer": [answer],
        }
        dataset = Dataset.from_dict(data)
        try:
            # Enforce 60 seconds hard timeout on Ragas evaluation
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(evaluate, dataset, metrics=[context_precision, context_recall])
                try:
                    res = future.result(timeout=60.0)
                except TimeoutError as exc:
                    raise TimeoutError(
                        f"Ragas evaluate call timed out after 60.0 seconds."
                    ) from exc
            try:
                if hasattr(res, "scores") and isinstance(res.scores, list) and len(res.scores) > 0:
                    return dict(res.scores[0])
                return dict(res)
            except Exception as exc:
                raise TypeError(
                    f"Ragas result conversion failed — check if Ragas API has changed. "
                    f"Could not convert {type(res)} to dict. Error: {exc}"
                ) from exc
        except Exception as e:
            if "Ragas result conversion failed" in str(e):
                raise
            logger.error("Ragas evaluation failed: %s", e)
            return {"context_precision": 0.0, "context_recall": 0.0}

    def _run_llm_judge(self, eval_input: EvaluationInput) -> dict[str, Any]:
        """Call LLM judge for custom noise/coverage rubric."""
        user_prompt = _build_user_prompt(eval_input)
        parsed, raw = self._judge.call_with_json(_SYSTEM_PROMPT, user_prompt)
        return parsed

    @staticmethod
    def _extract_score(parsed: dict[str, Any], key: str, default: int = 3) -> int:
        """Safely extract score from key dict structure."""
        entry = parsed.get(key)
        if isinstance(entry, dict):
            raw = entry.get("score", default)
        else:
            raw = entry if entry is not None else default

        try:
            score = int(raw)
        except (TypeError, ValueError):
            return default

        # Clamp to [1, 5]
        return max(1, min(score, 5))
