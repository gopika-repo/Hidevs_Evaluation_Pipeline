"""
Main orchestrator for Dave evaluation pipeline batch runs.

Executes evaluations across all dataset conversations, computes aggregated metrics,
and writes formatted JSON and CSV outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from evaluation_pipeline.data.dataset_builder import DatasetBuilder
from evaluation_pipeline.data.mock_conversations import get_mock_conversations

# Setup logger to output both to stderr and a dedicated file for visual tracking
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Stderr handler
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# File handler
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/pipeline_progress.log", mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Ingest Data (Phase 0A)
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("PHASE 0A — Ingesting conversations dataset")
    logger.info("-" * 70)

    raw_records = get_mock_conversations()
    logger.info("Loaded %d raw conversation records.", len(raw_records))

    # ------------------------------------------------------------------
    # 2. Tag & Validate (Phase 0A)
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("PHASE 0A — Validating and tagging dataset")
    logger.info("-" * 70)

    builder = DatasetBuilder()
    evaluation_inputs = builder.build(raw_records)
    logger.info(
        "Dataset construction complete. %d/%d records successfully built.",
        len(evaluation_inputs),
        len(raw_records),
    )

    if not evaluation_inputs:
        logger.error("No valid evaluation records found. Exiting.")
        return

    # Command line argument parser for full vs quick mode
    parser = argparse.ArgumentParser(description="Dave AI Assistant Evaluation Pipeline")
    parser.add_argument("--full", action="store_true", help="Run the full 23-conversation batch")
    args, unknown = parser.parse_known_args()

    full_mode = args.full or os.getenv("PIPELINE_MODE") == "full"
    if not full_mode:
        print("Running in QUICK mode (first 5 conversations). Use --full or PIPELINE_MODE=full for the entire batch.", flush=True)
        logger.info("Running in QUICK mode (first 5 conversations). Use --full or PIPELINE_MODE=full for the entire batch.")
        evaluation_inputs = evaluation_inputs[:5]
    else:
        print(f"Running in FULL mode (all {len(evaluation_inputs)} conversations).", flush=True)
        logger.info("Running in FULL mode (all %d conversations).", len(evaluation_inputs))

    _print_dataset_summary(evaluation_inputs)

    from evaluation_pipeline.evaluators.response_quality_evaluator import ResponseQualityEvaluator
    from evaluation_pipeline.evaluators.groundedness_evaluator import GroundednessEvaluator
    from evaluation_pipeline.evaluators.safety_evaluator import SafetyEvaluator
    from evaluation_pipeline.evaluators.intent_evaluator import IntentEvaluator
    from evaluation_pipeline.data.models import EvaluationResult

    rq_evaluator = ResponseQualityEvaluator()
    gd_evaluator = GroundednessEvaluator()
    safety_evaluator = SafetyEvaluator()
    intent_evaluator = IntentEvaluator()

    rq_results = []
    gd_results = []
    safety_results = []
    intent_results = []

    total_convs = len(evaluation_inputs)
    
    # Report realistic time estimate upfront
    avg_sec = 25.0  # default fallback
    stats_file = "logs/last_run_stats.json"
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r") as sf:
                stats = json.load(sf)
                avg_sec = stats.get("avg_seconds_per_convo", 25.0)
        except Exception:
            pass
    est_total_sec = avg_sec * total_convs
    est_min = est_total_sec / 60.0
    print(f"Estimated runtime: ~{est_min:.1f} minutes for {total_convs} conversations (based on average {avg_sec:.1f}s per conversation).", flush=True)
    logger.info("Estimated runtime: ~%.1f minutes for %d conversations.", est_min, total_convs)

    logger.info("-" * 70)
    logger.info("PHASE 1 — Running Batch Evaluations (conversation-by-conversation)")
    logger.info("-" * 70)

    batch_start_time = time.time()

    for idx, eval_input in enumerate(evaluation_inputs, start=1):
        conv_id = eval_input.conversation_id
        conv_start_time = time.time()
        msg_start = f"[{idx}/{total_convs}] Starting evaluation for conversation_id={conv_id}"
        print(msg_start, flush=True)
        logger.info("-" * 70)
        logger.info(msg_start)
        logger.info("-" * 70)

        # Run independent evaluators (RQ, Safety, Intent) in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_rq = executor.submit(rq_evaluator.evaluate, eval_input)
            future_sf = executor.submit(safety_evaluator.evaluate, eval_input)
            future_it = executor.submit(intent_evaluator.evaluate, eval_input)

            # 1. Response Quality
            logger.info("[%d/%d] Running ResponseQualityEvaluator...", idx, total_convs)
            try:
                rq_res = future_rq.result()
                logger.info("[%d/%d] ResponseQualityEvaluator finished for %s (score=%.2f/%.2f)", idx, total_convs, conv_id, rq_res.score, rq_res.max_score)
            except Exception as exc:
                logger.error("[%d/%d] ResponseQualityEvaluator failed for %s: %s", idx, total_convs, conv_id, exc, exc_info=True)
                rq_res = EvaluationResult(
                    evaluator_name=rq_evaluator.name,
                    conversation_id=conv_id,
                    score=0.0,
                    max_score=20.0,
                    sub_scores={},
                    feedback=f"Evaluation failed with error: {exc}",
                    flagged=True,
                )
            rq_results.append(rq_res)

            # 2. Safety
            logger.info("[%d/%d] Running SafetyEvaluator...", idx, total_convs)
            try:
                safety_res = future_sf.result()
                logger.info("[%d/%d] SafetyEvaluator finished for %s (score=%.2f/%.2f)", idx, total_convs, conv_id, safety_res.score, safety_res.max_score)
            except Exception as exc:
                logger.error("[%d/%d] SafetyEvaluator failed for %s: %s", idx, total_convs, conv_id, exc, exc_info=True)
                safety_res = EvaluationResult(
                    evaluator_name=safety_evaluator.name,
                    conversation_id=conv_id,
                    score=0.0,
                    max_score=15.0,
                    sub_scores={},
                    feedback=f"Evaluation failed with error: {exc}",
                    flagged=True,
                )
            safety_results.append(safety_res)

            # 3. Intent Understanding
            logger.info("[%d/%d] Running IntentEvaluator...", idx, total_convs)
            try:
                intent_res = future_it.result()
                logger.info("[%d/%d] IntentEvaluator finished for %s (score=%.2f/%.2f)", idx, total_convs, conv_id, intent_res.score, intent_res.max_score)
            except Exception as exc:
                logger.error("[%d/%d] IntentEvaluator failed for %s: %s", idx, total_convs, conv_id, exc, exc_info=True)
                intent_res = EvaluationResult(
                    evaluator_name=intent_evaluator.name,
                    conversation_id=conv_id,
                    score=0.0,
                    max_score=15.0,
                    sub_scores={},
                    feedback=f"Evaluation failed with error: {exc}",
                    flagged=True,
                )
            intent_results.append(intent_res)

        # 4. Groundedness (sequential)
        logger.info("[%d/%d] Running GroundednessEvaluator...", idx, total_convs)
        try:
            gd_res = gd_evaluator.evaluate(eval_input)
            logger.info("[%d/%d] GroundednessEvaluator finished for %s (score=%.2f/%.2f)", idx, total_convs, conv_id, gd_res.score, gd_res.max_score)
        except Exception as exc:
            logger.error("[%d/%d] GroundednessEvaluator failed for %s: %s", idx, total_convs, conv_id, exc, exc_info=True)
            gd_res = EvaluationResult(
                evaluator_name=gd_evaluator.name,
                conversation_id=conv_id,
                score=0.0,
                max_score=15.0,
                sub_scores={},
                feedback=f"Evaluation failed with error: {exc}",
                flagged=True,
            )
        gd_results.append(gd_res)

        elapsed = time.time() - conv_start_time
        msg_finish = f"[{idx}/{total_convs}] Finished evaluation for conversation_id={conv_id} in {elapsed:.2f}s"
        print(msg_finish, flush=True)
        logger.info(msg_finish)

    batch_elapsed = time.time() - batch_start_time
    logger.info("=" * 70)
    logger.info("BATCH COMPLETE: Processed %d/%d conversations in %.2fs", total_convs, total_convs, batch_elapsed)
    logger.info("=" * 70)

    # Save run statistics for future runtime estimates
    avg_seconds = batch_elapsed / total_convs if total_convs > 0 else 0.0
    try:
        with open("logs/last_run_stats.json", "w") as sf:
            json.dump({"avg_seconds_per_convo": avg_seconds, "last_run_time": batch_elapsed}, sf)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5. Aggregate Scores (Phase 0C)
    # ------------------------------------------------------------------
    from evaluation_pipeline.aggregator.score_aggregator import (
        ScoreAggregator,
    )

    logger.info("-" * 70)
    logger.info("PHASE 0C — Aggregating Scores")
    logger.info("-" * 70)

    aggregator = ScoreAggregator()
    aggregated_data = aggregator.aggregate_dataset(
        evaluation_inputs,
        rq_results,
        gd_results,
        safety_results,
        intent_results,
    )

    # ------------------------------------------------------------------
    # 6. Generate Output Reports (Phase 0C)
    # ------------------------------------------------------------------
    from evaluation_pipeline.output.report_generator import (
        ReportGenerator,
    )

    logger.info("-" * 70)
    logger.info("PHASE 0C — Generating Output Reports")
    logger.info("-" * 70)

    generator = ReportGenerator()
    generator.generate_reports(aggregated_data)

    # ------------------------------------------------------------------
    # 7. Print results tables
    # ------------------------------------------------------------------
    print("\n")
    _print_results_table("RESPONSE QUALITY RESULTS (max=20)", rq_results)
    _print_results_table("GROUNDEDNESS RESULTS (max=20)", gd_results)
    _print_results_table("SAFETY RESULTS (max=20)", safety_results)
    _print_results_table("INTENT UNDERSTANDING RESULTS (max=20)", intent_results)
    _print_health_table("OVERALL HEALTH SUMMARY (Phase 1, 0-80 scale)", aggregated_data["conversations"])

    # ------------------------------------------------------------------
    # 8. Detailed feedback inspection (3+ samples)
    # ------------------------------------------------------------------
    _print_detailed_inspection(
        evaluation_inputs,
        rq_results,
        gd_results,
        safety_results,
        intent_results,
        aggregated_data["conversations"]
    )

    # ------------------------------------------------------------------
    # 9. Summary statistics
    # ------------------------------------------------------------------
    _print_summary_stats(rq_results, gd_results, safety_results, intent_results, aggregated_data)

    logger.info(
        "Phase 1 complete. Total evaluation time: %.1fs", batch_elapsed
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_dataset_summary(inputs: list) -> None:
    """Print high-level dataset summary."""
    from evaluation_pipeline.data.models import ConversationType

    cb = sum(1 for i in inputs if i.conversation_type == ConversationType.CONTEXT_BACKED)
    cf = sum(1 for i in inputs if i.conversation_type == ConversationType.CONTEXT_FREE)

    print("\n" + "=" * 70)
    print("  DATASET SUMMARY")
    print("=" * 70)
    print(f"  Total conversations:  {len(inputs)}")
    print(f"  Context-backed:       {cb}")
    print(f"  Context-free:         {cf}")
    print("=" * 70 + "\n")


def _print_results_table(title: str, results: list) -> None:
    """Print a compact results table."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print(
        f"  {'ID':<10} {'Score':>8} {'Max':>6} {'%':>7} "
        f"{'Flagged':<8} {'Sub-Scores'}"
    )
    print("  " + "-" * 86)

    for r in results:
        if r.score is None:
            pct_str = "  N/A%"
            score_str = "     N/A"
        else:
            pct = (r.score / r.max_score * 100) if r.max_score > 0 else 0
            pct_str = f"{pct:>6.1f}%"
            score_str = f"{r.score:>8.2f}"

        flag_str = "! YES" if r.flagged else "  no"
        sub_str = ", ".join(f"{k}={v}" for k, v in r.sub_scores.items())
        # Truncate sub-scores display if too long
        if len(sub_str) > 50:
            sub_str = sub_str[:47] + "..."
        print(
            f"  {r.conversation_id:<10} {score_str} "
            f"{r.max_score:>6.0f} {pct_str} "
            f"{flag_str:<8} {sub_str}"
        )

    print()


def _print_health_table(title: str, records: list) -> None:
    """Print overall health scores summary."""
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print(
        f"  {'ID':<10} {'Type':<15} {'Health Score':>14} {'Flagged':<8}"
    )
    print("  " + "-" * 86)

    for r in records:
        flag_str = "! YES" if r["flagged"] else "  no"
        print(
            f"  {r['conversation_id']:<10} {r['conversation_type']:<15} "
            f"{r['overall_health_score']:>14.2f} {flag_str:<8}"
        )

    print()


def _print_detailed_inspection(
    inputs: list,
    rq_results: list,
    gd_results: list,
    safety_results: list,
    intent_results: list,
    records: list,
) -> None:
    """
    Print detailed feedback for sample conversations to verify
    the LLM is generating conversation-specific reasoning.
    """
    inspect_ids = ["CB-001", "KI-002", "KI-003", "CF-001"]

    rq_by_id = {r.conversation_id: r for r in rq_results}
    gd_by_id = {r.conversation_id: r for r in gd_results}
    safety_by_id = {r.conversation_id: r for r in safety_results}
    intent_by_id = {r.conversation_id: r for r in intent_results}
    record_by_id = {r["conversation_id"]: r for r in records}

    print("=" * 90)
    print("  DETAILED FEEDBACK INSPECTION (verify LLM generates specific reasoning)")
    print("=" * 90)

    for conv_id in inspect_ids:
        rq = rq_by_id.get(conv_id)
        gd = gd_by_id.get(conv_id)
        safety = safety_by_id.get(conv_id)
        rec = record_by_id.get(conv_id)

        if not rq or not gd or not safety or not rec:
            continue

        print(f"\n{'-' * 90}")
        print(f"  CONVERSATION: {conv_id} | Health Score: {rec['overall_health_score']:.2f}")
        print(f"{'-' * 90}")

        # Find the input for context
        inp = next((i for i in inputs if i.conversation_id == conv_id), None)
        if inp:
            print(f"  Type: {inp.conversation_type.value}")
            print(f"  Query: {inp.user_query[:80]}...")

        print(f"\n  -- Response Quality ({rq.score}/{rq.max_score}) "
              f"{'! FLAGGED' if rq.flagged else ''} --")
        for line in rq.feedback.split("\n"):
            print(f"  {line}")

        print(f"\n  -- Groundedness ({gd.score}/{gd.max_score}) "
              f"{'! FLAGGED' if gd.flagged else ''} --")
        for line in gd.feedback.split("\n"):
            print(f"  {line}")

        print(f"\n  -- Safety ({safety.score}/{safety.max_score}) "
              f"{'! FLAGGED' if safety.flagged else ''} --")
        for line in safety.feedback.split("\n"):
            print(f"  {line}")

        intent = intent_by_id.get(conv_id)
        if intent:
            print(f"\n  -- Intent Understanding ({intent.score}/{intent.max_score}) "
                  f"{'! FLAGGED' if intent.flagged else ''} --")
            for line in intent.feedback.split("\n"):
                print(f"  {line}")

    print()


def _print_summary_stats(
    rq_results: list,
    gd_results: list,
    safety_results: list,
    intent_results: list,
    aggregated_data: dict,
) -> None:
    """Print aggregate statistics across all evaluations."""
    stats = aggregated_data["summary_stats"]
    averages = stats["averages"]
    breakdown = stats["groundedness_breakdown"]

    print("=" * 70)
    print("  AGGREGATE STATISTICS")
    print("=" * 70)
    print(f"  Total conversations:  {stats['total_conversations']}")
    print(f"  Flagged:              {stats['flagged_conversations']}")
    print(f"  Averages:")
    print(f"    Response Quality:   {averages['response_quality']:.2f} / 20.00")
    print(f"    Groundedness:       {averages['groundedness']:.2f} / 20.00")
    print(f"    Safety:             {averages['safety']:.2f} / 20.00")
    if "intent_understanding" in averages:
        print(f"    Intent Understand:  {averages['intent_understanding']:.2f} / 20.00")
    print(f"    Overall Health:     {averages['overall_health']:.2f} / 80.00  (Phase 1 scope)")
    print(f"  Groundedness Breakdown:")
    print(f"    Context-backed avg: {breakdown['context_backed_average']:.2f} (count: {breakdown['context_backed_count']})")
    print(f"    Context-free avg:   {breakdown['context_free_average']:.2f} (count: {breakdown['context_free_count']})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
