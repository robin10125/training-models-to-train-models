from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np


def result_sort_key(result: Any) -> float:
    score = result_score(result)
    if score is None:
        return float("-inf")
    return score


def result_score(result: Any) -> float | None:
    if result.verified_score is not None:
        return float(result.verified_score)
    metadata = result.metadata or {}
    if metadata.get("verified_score") is not None:
        return float(metadata["verified_score"])
    if result.rlvr_reward is not None and result.status == "success":
        return float(result.rlvr_reward)
    return result.mean_reward


def elite_context_from_results(results: list[Any], elite_count: int) -> list[dict[str, Any]]:
    if elite_count < 1:
        raise ValueError("--eureka-elites must be at least 1")
    elites = sorted(results, key=result_sort_key, reverse=True)[:elite_count]
    return [
        {
            "rank": rank,
            "name": result.candidate.name,
            "expression": result.candidate.expression,
            "score": result_score(result),
            "mean_reward": result.mean_reward,
            "std_reward": result.std_reward,
            "status": result.status,
            "verified_reward_type": result.verified_reward_type,
        }
        for rank, result in enumerate(elites, start=1)
    ]


def annotate_generation_results(results: list[Any], elite_count: int) -> list[Any]:
    annotated = []
    generation_size = len(results)
    for rank, result in enumerate(results, start=1):
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "eureka_generation_rank": rank,
                "eureka_generation_size": generation_size,
                "eureka_selected_elite": rank <= elite_count,
            }
        )
        annotated.append(replace(result, metadata=metadata))
    return annotated


def negative_reward_for_generation(results: list[Any], margin: float) -> float:
    if margin <= 0:
        raise ValueError("--negative-rlvr-margin must be greater than 0")
    successful_scores = [
        result_score(result)
        for result in results
        if result.status == "success" and result_score(result) is not None
    ]
    baseline = min(successful_scores) if successful_scores else 0.0
    return float(baseline) - margin


def assign_failed_evaluation_penalties(results: list[Any], penalty: float) -> list[Any]:
    return [
        replace(result, rlvr_reward=penalty, rlvr_reward_type="failed_evaluation_penalty")
        if result.status == "failed"
        else result
        for result in results
    ]


def format_generation_feedback(results: list[Any], elite_count: int) -> str:
    ranked = sorted(results, key=result_sort_key, reverse=True)
    elites = ranked[:elite_count]
    rejected = ranked[elite_count:]
    successful_rewards = [result_score(result) for result in ranked if result_score(result) is not None]
    lines = ["Ranked by conservative verified score: mean_return - 0.25 * std_return."]
    if successful_rewards:
        lines.append(
            "Verified return summary: "
            f"best={max(successful_rewards):.4f}, "
            f"mean={float(np.mean(successful_rewards)):.4f}, "
            f"worst={min(successful_rewards):.4f}."
        )
    lines.append("Elite reward programs to improve from:")
    for rank, result in enumerate(elites, start=1):
        lines.extend(format_result_feedback_row(rank, result, selected=True))
    if rejected:
        lines.append("Lower-ranked reward programs to avoid copying blindly:")
        for rank, result in enumerate(rejected[: min(4, len(rejected))], start=elite_count + 1):
            lines.extend(format_result_feedback_row(rank, result, selected=False))
    failure_count = sum(1 for result in ranked if result.status == "failed")
    if failure_count:
        lines.append(f"{failure_count} candidates failed validation or evaluation; avoid their error patterns.")
    return "\n".join(lines)


def format_result_feedback_row(rank: int, result: Any, *, selected: bool) -> list[str]:
    score_value = result_score(result)
    score = "n/a" if score_value is None else f"{score_value:.4f}"
    mean = "n/a" if result.mean_reward is None else f"{result.mean_reward:.4f}"
    std = "n/a" if result.std_reward is None else f"{result.std_reward:.4f}"
    prefix = "ELITE" if selected else "NON_ELITE"
    lines = [
        f"{rank}. {prefix} name={result.candidate.name} status={result.status} score={score} mean_return={mean} std={std}",
        f"   expression={result.candidate.expression}",
    ]
    metadata = result.metadata or {}
    best_shaped = metadata.get("best_shaped_return")
    summaries = metadata.get("iteration_summaries") or []
    true_values = [
        item.get("best_true_return_in_population")
        for item in summaries
        if item.get("best_true_return_in_population") is not None
    ]
    diagnostics = []
    if best_shaped is not None:
        diagnostics.append(f"best_shaped_return={float(best_shaped):.4f}")
    if true_values:
        diagnostics.append(f"best_internal_true_return={max(float(value) for value in true_values):.4f}")
    if result.error:
        diagnostics.append(f"error={result.error.splitlines()[0]}")
    if diagnostics:
        lines.append("   diagnostics=" + ", ".join(diagnostics))
    component_stats = latest_component_stats(metadata)
    if component_stats:
        parts = [
            f"{name}:mean={float(stats['mean']):.4f},min={float(stats['min']):.4f},max={float(stats['max']):.4f}"
            for name, stats in component_stats.items()
        ]
        lines.append("   reward_component_stats=" + "; ".join(parts))
    return lines


def latest_component_stats(metadata: dict[str, Any]) -> dict[str, dict[str, float]]:
    summaries = metadata.get("iteration_summaries") or []
    for item in reversed(summaries):
        stats = item.get("reward_component_stats")
        if stats:
            return stats
    return {}
