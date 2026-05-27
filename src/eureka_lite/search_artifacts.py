from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_text, write_jsonl
from .rewards import RewardCandidate
from .search_feedback import result_sort_key
from .search_state import GenerationPhase, GenerationState, SearchState
from .search_types import CandidateResult, RunConfig


def write_results(output_dir: Path, results: list[CandidateResult]) -> None:
    payload = [candidate_result_to_dict(result) for result in sorted(results, key=result_sort_key, reverse=True)]
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "rlvr_records.jsonl", [to_rlvr_record(result) for result in results])
    best = next((row for row in payload if row["status"] != "invalid_completion"), None)
    if best is not None:
        (output_dir / "best_reward.py").write_text(
            "# Best discovered reward expression\n"
            f"REWARD_EXPRESSION = {best['candidate']['expression']!r}\n"
            f"MEAN_REWARD = {best['mean_reward']!r}\n",
            encoding="utf-8",
        )


def to_rlvr_record(result: CandidateResult) -> dict[str, object]:
    candidate = result.candidate
    return {
        "prompt_id": candidate.prompt_id,
        "generator_type": candidate.generator_type,
        "generator_checkpoint": candidate.generator_checkpoint,
        "prompt": candidate.prompt,
        "completion": candidate.completion_text or candidate.expression,
        "completion_token_ids": candidate.completion_token_ids,
        "old_logprobs": candidate.old_logprobs,
        "task": result.task,
        "candidate_name": candidate.name,
        "candidate_generation": candidate.generation,
        "eureka_role": candidate.eureka_role,
        "eureka_parent_names": candidate.eureka_parent_names,
        "eureka_parent_expressions": candidate.eureka_parent_expressions,
        "eureka_parent_scores": candidate.eureka_parent_scores,
        "eureka_elite_names": candidate.eureka_elite_names,
        "eureka_elite_expressions": candidate.eureka_elite_expressions,
        "eureka_elite_scores": candidate.eureka_elite_scores,
        "eureka_feedback": candidate.eureka_feedback,
        "train_reward_expression": None if result.status == "invalid_completion" else candidate.expression,
        "reward_components": candidate.component_expressions,
        "verified_reward": result.mean_reward,
        "verified_reward_std": result.std_reward,
        "verified_reward_episodes": result.episode_rewards,
        "verified_reward_type": result.verified_reward_type,
        "rlvr_reward": result.mean_reward if result.rlvr_reward is None else result.rlvr_reward,
        "rlvr_reward_type": result.verified_reward_type if result.rlvr_reward_type is None else result.rlvr_reward_type,
        "seed": result.seed,
        "timesteps": result.timesteps,
        "status": result.status,
        "error": result.error,
        "elapsed_seconds": result.elapsed_seconds,
        "metadata": result.metadata,
    }


def prepare_output_dir(output_dir: Path, config: RunConfig, *, resume: bool, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    checkpoint_path = output_dir / "checkpoint.json"
    if overwrite and not resume:
        for path in (
            "results.json",
            "rlvr_records.jsonl",
            "rlvr_records.incremental.jsonl",
            "events.jsonl",
            "run.log",
            "checkpoint.json",
            "best_reward.py",
            "run_config.json",
        ):
            target = output_dir / path
            if target.exists():
                target.unlink()
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume: missing {checkpoint_path}")
        return
    if config_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains a run_config.json. Use --resume to continue or --overwrite to replace."
        )
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def save_checkpoint(
    output_dir: Path,
    config: RunConfig,
    *,
    results: list[CandidateResult],
    next_generation: int,
    next_candidates: list[RewardCandidate],
    best_expression: str | None,
    best_score: float | None,
    elite_context: list[dict[str, Any]] | None = None,
    evolution_feedback: str | None = None,
    search_state: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": 2 if search_state is not None else 1,
        "config": asdict(config),
        "next_generation": next_generation,
        "next_candidates": [asdict(candidate) for candidate in next_candidates],
        "best_expression": best_expression,
        "best_score": best_score,
        "elite_context": elite_context or [],
        "evolution_feedback": evolution_feedback,
        "results": [candidate_result_to_dict(result) for result in results],
        "updated_at": time.time(),
    }
    if search_state is not None:
        payload["search_state"] = search_state
    atomic_write_text(output_dir / "checkpoint.json", json.dumps(payload, indent=2))


def load_checkpoint(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "checkpoint.json").read_text(encoding="utf-8"))


def persist_search_state(output_dir: Path, config: RunConfig, state: SearchState) -> None:
    current = state.generation
    resume_results = state.finalized_results + ([] if current is None else current.raw_results)
    save_checkpoint(
        output_dir,
        config,
        results=resume_results,
        next_generation=config.generations if current is None else current.index,
        next_candidates=[] if current is None else current.candidates,
        best_expression=state.best_expression,
        best_score=state.best_score,
        elite_context=state.elite_context,
        evolution_feedback=state.evolution_feedback,
        search_state=search_state_to_dict(state),
    )


def search_state_to_dict(state: SearchState) -> dict[str, Any]:
    current = state.generation
    generation = None
    if current is not None:
        generation = {
            "index": current.index,
            "phase": current.phase.value,
            "candidates": [asdict(candidate) for candidate in current.candidates],
            "raw_results": [candidate_result_to_dict(result) for result in current.raw_results],
            "rejected_candidates": [
                {"candidate": asdict(candidate), "error": error}
                for candidate, error in current.rejected_candidates
            ],
        }
    return {
        "finalized_results": [candidate_result_to_dict(result) for result in state.finalized_results],
        "generation": generation,
        "best_expression": state.best_expression,
        "best_score": state.best_score,
        "elite_context": state.elite_context,
        "evolution_feedback": state.evolution_feedback,
        "completed": state.completed,
    }


def restore_search_state(checkpoint: dict[str, Any], config: RunConfig) -> SearchState:
    payload = checkpoint.get("search_state")
    if payload is not None:
        generation_row = payload.get("generation")
        generation = None
        if generation_row is not None:
            generation = GenerationState(
                index=int(generation_row["index"]),
                phase=GenerationPhase(generation_row["phase"]),
                candidates=[candidate_from_dict(row) for row in generation_row.get("candidates", [])],
                raw_results=[candidate_result_from_dict(row) for row in generation_row.get("raw_results", [])],
                rejected_candidates=[
                    (candidate_from_dict(row["candidate"]), row["error"])
                    for row in generation_row.get("rejected_candidates", [])
                ],
            )
        return SearchState(
            finalized_results=[candidate_result_from_dict(row) for row in payload.get("finalized_results", [])],
            generation=generation,
            best_expression=payload.get("best_expression"),
            best_score=payload.get("best_score"),
            elite_context=list(payload.get("elite_context", [])),
            evolution_feedback=payload.get("evolution_feedback"),
            completed=bool(payload.get("completed", False)),
        )
    results = [candidate_result_from_dict(row) for row in checkpoint.get("results", [])]
    next_generation = int(checkpoint.get("next_generation", 0))
    state = SearchState(
        best_expression=checkpoint.get("best_expression"),
        best_score=checkpoint.get("best_score"),
        elite_context=list(checkpoint.get("elite_context", [])),
        evolution_feedback=checkpoint.get("evolution_feedback"),
    )
    if next_generation >= config.generations:
        state.finalized_results = results
        state.completed = True
        return state
    next_candidates = [candidate_from_dict(row) for row in checkpoint.get("next_candidates", [])]
    if not next_candidates:
        state.finalized_results = results
        state.generation = GenerationState(index=next_generation, phase=GenerationPhase.NEEDS_POPULATION)
        return state
    current_names = {candidate.name for candidate in next_candidates}
    state.generation = GenerationState(
        index=next_generation,
        phase=GenerationPhase.EVALUATING,
        candidates=next_candidates,
        raw_results=[
            result
            for result in results
            if result.candidate.generation == next_generation and result.candidate.name in current_names
        ],
    )
    state.finalized_results = [result for result in results if result not in state.generation.raw_results]
    return state


def candidate_result_to_dict(result: CandidateResult) -> dict[str, Any]:
    row = asdict(result)
    row["candidate"] = asdict(result.candidate)
    return row


def candidate_result_from_dict(row: dict[str, Any]) -> CandidateResult:
    return CandidateResult(
        candidate=candidate_from_dict(row["candidate"]),
        mean_reward=row["mean_reward"],
        std_reward=row["std_reward"],
        episode_rewards=list(row["episode_rewards"]),
        timesteps=int(row["timesteps"]),
        seed=int(row["seed"]),
        task=row["task"],
        verified_reward_type=row.get("verified_reward_type", "true_env_return"),
        rlvr_reward=row.get("rlvr_reward"),
        rlvr_reward_type=row.get("rlvr_reward_type"),
        status=row.get("status", "success"),
        error=row.get("error"),
        elapsed_seconds=row.get("elapsed_seconds"),
        metadata=row.get("metadata"),
    )


def candidate_from_dict(row: dict[str, Any]) -> RewardCandidate:
    return RewardCandidate(
        name=row["name"],
        task=row["task"],
        prompt_id=row["prompt_id"],
        prompt=row["prompt"],
        expression=row["expression"],
        weights=dict(row.get("weights", {})),
        component_expressions=row.get("component_expressions"),
        completion_text=row.get("completion_text"),
        generation=int(row.get("generation", 0)),
        generator_type=row.get("generator_type", "mock"),
        generator_checkpoint=row.get("generator_checkpoint", "mock-v1"),
        completion_token_ids=row.get("completion_token_ids"),
        old_logprobs=row.get("old_logprobs"),
        eureka_role=row.get("eureka_role", "initial"),
        eureka_parent_names=row.get("eureka_parent_names"),
        eureka_parent_expressions=row.get("eureka_parent_expressions"),
        eureka_parent_scores=row.get("eureka_parent_scores"),
        eureka_elite_names=row.get("eureka_elite_names"),
        eureka_elite_expressions=row.get("eureka_elite_expressions"),
        eureka_elite_scores=row.get("eureka_elite_scores"),
        eureka_feedback=row.get("eureka_feedback"),
    )


def result_key(generation: int, candidate_name: str) -> str:
    return f"{generation}:{candidate_name}"


def log_event(output_dir: Path, event: str, payload: dict[str, Any]) -> None:
    row = {"time": time.time(), "event": event, **payload}
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def log_message(output_dir: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (output_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
