from __future__ import annotations

import random
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import get_adapter
from .generators import initial_population, mutate_candidate
from .hf_generator import HfGeneratorConfig, HfRewardGenerator
from .rewards import RewardCandidate, RewardWrapper
from .search_artifacts import (
    candidate_from_dict,
    candidate_result_from_dict,
    candidate_result_to_dict,
    load_checkpoint,
    log_event,
    log_message,
    persist_search_state,
    prepare_output_dir,
    result_key,
    restore_search_state,
    save_checkpoint,
    to_rlvr_record,
    write_results,
)
from .search_feedback import (
    annotate_generation_results,
    assign_failed_evaluation_penalties,
    elite_context_from_results,
    format_generation_feedback,
    negative_reward_for_generation,
    result_score,
    result_sort_key,
)
from .search_state import GenerationPhase, GenerationState, SearchState
from .search_types import CandidateEvaluationConfig, CandidateResult, RunConfig


def train_and_evaluate(
    candidate: RewardCandidate,
    config: CandidateEvaluationConfig,
) -> CandidateResult:
    started_at = time.monotonic()
    adapter = get_adapter(config.task)
    if config.timesteps <= 0:
        return CandidateResult(
            candidate=candidate,
            mean_reward=None,
            std_reward=None,
            episode_rewards=[],
            timesteps=config.timesteps,
            seed=config.seed,
            task=config.task,
            status="generated_only",
            elapsed_seconds=time.monotonic() - started_at,
        )

    if config.sim_backend == "mjwarp":
        from .mjwarp_evaluator import train_and_evaluate_mjwarp

        result = train_and_evaluate_mjwarp(
            candidate,
            mjwarp_evaluator_config(config),
        )
        return CandidateResult(
            candidate=candidate,
            mean_reward=result["mean_reward"],
            std_reward=result["std_reward"],
            episode_rewards=result["episode_rewards"],
            timesteps=int(result["metadata"]["training_world_steps"]),
            seed=config.seed,
            task=config.task,
            verified_reward_type=verified_reward_type_for_evaluator(config.mjwarp_verified_evaluator),
            rlvr_reward=result["verified_score"],
            rlvr_reward_type="conservative_verified_return",
            verified_score=result["verified_score"],
            elapsed_seconds=result["elapsed_seconds"],
            metadata=result["metadata"],
        )

    if config.sim_backend != "sb3":
        raise ValueError(f"Unsupported simulation backend: {config.sim_backend}")

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    def make_train_env():
        return RewardWrapper(gym.make(config.task), candidate.expression, adapter)

    train_env = make_vec_env(make_train_env, n_envs=config.n_envs, seed=config.seed)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=6,
        gamma=0.99,
        verbose=0,
        seed=config.seed,
        device=config.device,
    )
    model.learn(total_timesteps=config.timesteps, progress_bar=False)
    train_env.close()

    episode_rewards: list[float] = []
    for episode in range(config.eval_episodes):
        env = gym.make(config.task)
        obs, _info = env.reset(seed=config.seed + 10_000 + episode)
        total_reward = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
        env.close()
        episode_rewards.append(total_reward)

    stats = conservative_return_stats(episode_rewards)
    return CandidateResult(
        candidate=candidate,
        mean_reward=stats["mean_reward"],
        std_reward=stats["std_reward"],
        episode_rewards=episode_rewards,
        timesteps=config.timesteps,
        seed=config.seed,
        task=config.task,
        verified_reward_type="gym_ant_v5_return",
        rlvr_reward=stats["verified_score"],
        rlvr_reward_type="conservative_verified_return",
        verified_score=stats["verified_score"],
        elapsed_seconds=time.monotonic() - started_at,
        metadata={
            "verified_score": stats["verified_score"],
            "verified_score_std_weight": stats["verified_score_std_weight"],
            "verified_return_stats": stats,
            "common_eval_seed_start": config.seed + 10_000,
            "common_eval_seed_count": config.eval_episodes,
        },
    )


def conservative_return_stats(episode_rewards: list[float], *, std_weight: float = 0.25) -> dict[str, float]:
    values = np.asarray(episode_rewards, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "mean_reward": mean,
        "std_reward": std,
        "verified_score": mean - std_weight * std,
        "verified_score_std_weight": std_weight,
        "min_reward": float(np.min(values)),
        "p25_reward": float(np.percentile(values, 25)),
        "median_reward": float(np.median(values)),
        "p75_reward": float(np.percentile(values, 75)),
        "max_reward": float(np.max(values)),
    }


def mjwarp_evaluator_config(config: CandidateEvaluationConfig) -> Any:
    from .mjwarp_evaluator import MjwarpEvaluatorConfig

    options = config.mjwarp_options()
    warp_device = "cuda:0" if config.device in {"auto", "cuda"} else config.device
    return MjwarpEvaluatorConfig(
        task=config.task,
        evaluator=options.evaluator,
        worlds_per_candidate=config.worlds_per_candidate,
        episode_steps=options.episode_steps,
        training_episode_horizon=options.training_episode_horizon,
        policy_iterations=options.policy_iterations,
        ppo_horizon=options.ppo_horizon,
        ppo_epochs=options.ppo_epochs,
        ppo_minibatch_size=options.ppo_minibatch_size,
        ppo_learning_rate=options.ppo_learning_rate,
        ppo_init_mode=options.ppo_init_mode,
        base_policy_checkpoint=options.base_policy_checkpoint,
        elite_frac=options.elite_frac,
        rollout_mode=options.rollout_mode,
        verified_evaluator=options.verified_evaluator,
        verification_steps=options.verification_steps,
        verified_audit_gym=options.verified_audit_gym,
        verified_audit_max_abs_diff=options.verified_audit_max_abs_diff,
        reward_backend=options.reward_backend,
        use_cuda_graph=options.cuda_graph,
        seed=config.seed,
        device=warp_device,
        eval_episodes=config.eval_episodes,
    )


def verified_reward_type_for_evaluator(evaluator: str) -> str:
    if evaluator == "mjwarp":
        return "mjwarp_ant_return"
    if evaluator == "gym":
        return "gym_ant_v5_return"
    raise ValueError(f"Unsupported verified evaluator: {evaluator}")


def run_search(
    config: RunConfig,
    *,
    output_dir: Path,
    pause_path: Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> list[CandidateResult]:
    checkpoint = load_checkpoint(output_dir) if resume else None
    if checkpoint:
        config = RunConfig.from_dict(checkpoint["config"])
    rng = random.Random(config.seed)
    prepare_output_dir(output_dir, config, resume=resume, overwrite=overwrite)
    get_adapter(config.task)
    if config.generator not in {"mock", "hf"}:
        raise ValueError(f"Unsupported generator {config.generator!r}")
    hf_generator = make_hf_generator(config) if config.generator == "hf" else None
    search_state = restore_search_state(checkpoint, config) if checkpoint else SearchState(
        generation=GenerationState(index=0, phase=GenerationPhase.NEEDS_POPULATION)
    )
    if search_state.completed or search_state.generation is None:
        log_event(output_dir, "resume_noop", {"next_generation": config.generations, "generations": config.generations})
        log_message(output_dir, f"resume noop: next_generation={config.generations} generations={config.generations}")
        write_results(output_dir, search_state.finalized_results)
        return search_state.finalized_results

    log_event(output_dir, "run_started", {"config": asdict(config), "resume": resume})
    log_message(
        output_dir,
        f"run started: task={config.task} generator={config.generator} generations={config.generations} "
        f"population={config.population} sim_backend={config.sim_backend}",
    )
    while search_state.generation is not None:
        current = search_state.generation
        generation = current.index
        if current.phase == GenerationPhase.NEEDS_POPULATION:
            current.candidates, current.rejected_candidates = generate_population(
                config, generation, search_state, rng, hf_generator
            )
            current.phase = GenerationPhase.EVALUATING
            persist_search_state(output_dir, config, search_state)

        log_event(output_dir, "generation_started", {"generation": generation})
        log_message(output_dir, f"generation {generation} started")
        completed_keys = {
            result_key(result.candidate.generation, result.candidate.name) for result in current.raw_results
        }
        pending_candidates = []
        for candidate in current.candidates:
            if result_key(generation, candidate.name) in completed_keys:
                log_event(output_dir, "candidate_skipped", {"generation": generation, "candidate": candidate.name})
                log_message(output_dir, f"generation {generation} candidate {candidate.name} skipped")
                continue
            log_event(output_dir, "candidate_started", {"generation": generation, "candidate": candidate.name})
            log_message(output_dir, f"generation {generation} candidate {candidate.name} started")
            pending_candidates.append(candidate)
        evaluation_config = config.evaluation_config(config.seed + generation * 100)
        evaluation_groups = (
            [pending_candidates]
            if candidates_can_be_batched(pending_candidates, evaluation_config)
            else [[candidate] for candidate in pending_candidates]
        )
        for candidate_group in evaluation_groups:
            pending_results = run_candidates_safely(candidate_group, evaluation_config)
            current.raw_results.extend(pending_results)
            for candidate, result in zip(candidate_group, pending_results, strict=True):
                log_event(
                    output_dir,
                    "candidate_finished",
                    {
                        "generation": generation,
                        "candidate": candidate.name,
                        "status": result.status,
                        "mean_reward": result.mean_reward,
                        "std_reward": result.std_reward,
                        "verified_score": result_score(result),
                        "elapsed_seconds": result.elapsed_seconds,
                    },
                )
                log_message(
                    output_dir,
                    f"generation {generation} candidate {candidate.name} finished: "
                    f"status={result.status} mean_reward={result.mean_reward} "
                    f"verified_score={result_score(result)} elapsed_seconds={result.elapsed_seconds}",
                )
            # A GPU batch is a single durable evaluation transaction.
            persist_search_state(output_dir, config, search_state)
            if pause_requested(pause_path):
                last_candidate = candidate_group[-1]
                log_event(output_dir, "run_paused", {"generation": generation, "candidate": last_candidate.name})
                log_message(output_dir, f"run paused after generation {generation} candidate {last_candidate.name}")
                return search_state.finalized_results + current.raw_results

        generation_results, rejected_results = finalize_generation_results(current, config, output_dir)
        if not generation_results:
            search_state.finalized_results.extend(rejected_results)
            search_state.completed = True
            search_state.generation = None
            write_results(output_dir, search_state.finalized_results)
            persist_search_state(output_dir, config, search_state)
            log_event(output_dir, "generation_no_executable_candidates", {"generation": generation})
            log_message(output_dir, f"generation {generation} produced no executable reward candidates")
            return search_state.finalized_results

        search_state.finalized_results.extend(generation_results)
        search_state.finalized_results.extend(rejected_results)
        elite_count = min(config.eureka_elites, len(generation_results))
        search_state.elite_context = elite_context_from_results(generation_results, elite_count)
        search_state.evolution_feedback = format_generation_feedback(generation_results, elite_count)
        best = generation_results[0].candidate
        search_state.best_expression = best.expression
        search_state.best_score = result_score(generation_results[0])
        log_event(
            output_dir,
            "generation_finished",
            {
                "generation": generation,
                "best_candidate": best.name,
                "best_score": search_state.best_score,
                "elite_context": search_state.elite_context,
                "evolution_feedback": search_state.evolution_feedback,
            },
        )
        log_message(output_dir, f"generation {generation} finished: best={best.name} score={search_state.best_score}")
        write_results(output_dir, search_state.finalized_results)
        next_generation = generation + 1
        if next_generation >= config.generations:
            search_state.completed = True
            search_state.generation = None
        elif hf_generator is None:
            search_state.generation = GenerationState(
                index=next_generation,
                phase=GenerationPhase.EVALUATING,
                candidates=build_mock_offspring(
                    generation_results,
                    config,
                    next_generation,
                    rng,
                    search_state.elite_context,
                    search_state.evolution_feedback,
                ),
            )
        else:
            search_state.generation = GenerationState(index=next_generation, phase=GenerationPhase.NEEDS_POPULATION)
        persist_search_state(output_dir, config, search_state)
        if pause_requested(pause_path):
            log_event(output_dir, "run_paused", {"generation": generation})
            log_message(output_dir, f"run paused after generation {generation}")
            return search_state.finalized_results

    log_event(
        output_dir,
        "run_finished",
        {"results_count": len(search_state.finalized_results), "best_score": search_state.best_score},
    )
    log_message(
        output_dir,
        f"run finished: results_count={len(search_state.finalized_results)} best_score={search_state.best_score}",
    )
    return search_state.finalized_results


def make_hf_generator(config: RunConfig) -> HfRewardGenerator:
    return HfRewardGenerator(
        HfGeneratorConfig(
            model_id=config.model_id,
            adapter_path=config.adapter_path,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            load_in_4bit=config.load_in_4bit,
        )
    )


def generate_population(
    config: RunConfig,
    generation: int,
    state: SearchState,
    rng: random.Random,
    hf_generator: HfRewardGenerator | None,
) -> tuple[list[RewardCandidate], list[tuple[RewardCandidate, str]]]:
    if hf_generator is None:
        if generation != 0:
            raise RuntimeError("Mock offspring must be persisted at generation finalization")
        return initial_population(config.task, config.population, rng), []
    candidates = hf_generator.generate_population(
        task=config.task,
        population=config.population,
        generation=generation,
        best_expression=state.best_expression,
        best_score=state.best_score,
        elites=state.elite_context,
        evolution_feedback=state.evolution_feedback,
    )
    return candidates, hf_generator.drain_rejected_candidates()


def finalize_generation_results(
    current: GenerationState, config: RunConfig, output_dir: Path
) -> tuple[list[CandidateResult], list[CandidateResult]]:
    generation_results = list(current.raw_results)
    rejected_results: list[CandidateResult] = []
    if config.include_negative_rlvr_samples:
        failure_penalty = negative_reward_for_generation(generation_results, config.negative_rlvr_margin)
        generation_results = assign_failed_evaluation_penalties(generation_results, failure_penalty)
        rejected_results = rejected_candidates_to_results(
            current.rejected_candidates,
            task=config.task,
            seed=config.seed + current.index * 100,
            penalty=failure_penalty,
        )
        for result in rejected_results:
            log_event(
                output_dir,
                "candidate_rejected",
                {
                    "generation": current.index,
                    "candidate": result.candidate.name,
                    "status": result.status,
                    "rlvr_reward": result.rlvr_reward,
                    "error": result.error,
                },
            )
    if not generation_results:
        return [], rejected_results
    generation_results.sort(key=result_sort_key, reverse=True)
    return annotate_generation_results(generation_results, min(config.eureka_elites, len(generation_results))), rejected_results


def build_mock_offspring(
    generation_results: list[CandidateResult],
    config: RunConfig,
    generation: int,
    rng: random.Random,
    elite_context: list[dict[str, Any]],
    evolution_feedback: str | None,
) -> list[RewardCandidate]:
    parents = [
        result.candidate
        for result in generation_results[: max(1, min(config.eureka_elites, len(generation_results)))]
    ]
    parent_scores = {result.candidate.name: result_score(result) for result in generation_results}
    candidates = []
    for index in range(config.population):
        parent = parents[index % len(parents)]
        child = mutate_candidate(parent, index, generation, rng)
        candidates.append(
            replace(
                child,
                eureka_parent_scores=[parent_scores.get(parent.name)],
                eureka_elite_names=[str(item["name"]) for item in elite_context],
                eureka_elite_expressions=[str(item["expression"]) for item in elite_context],
                eureka_elite_scores=[item["score"] for item in elite_context],
                eureka_feedback=evolution_feedback,
            )
        )
    return candidates


def run_candidate_safely(
    candidate: RewardCandidate,
    config: CandidateEvaluationConfig,
) -> CandidateResult:
    try:
        return train_and_evaluate(candidate, config)
    except Exception as exc:
        return CandidateResult(
            candidate=candidate,
            mean_reward=None,
            std_reward=None,
            episode_rewards=[],
            timesteps=config.timesteps,
            seed=config.seed,
            task=config.task,
            status="failed",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            metadata={"sim_backend": config.sim_backend},
        )


def run_candidates_safely(
    candidates: list[RewardCandidate],
    config: CandidateEvaluationConfig,
) -> list[CandidateResult]:
    if not candidates:
        return []
    if not candidates_can_be_batched(candidates, config):
        return [run_candidate_safely(candidate, config) for candidate in candidates]
    try:
        from .mjwarp_evaluator import train_and_evaluate_mjwarp_batch

        rows = train_and_evaluate_mjwarp_batch(candidates, mjwarp_evaluator_config(config))
        return [
            CandidateResult(
                candidate=candidate,
                mean_reward=row["mean_reward"],
                std_reward=row["std_reward"],
                episode_rewards=row["episode_rewards"],
                timesteps=int(row["metadata"]["training_world_steps"]),
                seed=config.seed,
                task=config.task,
                verified_reward_type=verified_reward_type_for_evaluator(config.mjwarp_verified_evaluator),
                rlvr_reward=row["verified_score"],
                rlvr_reward_type="conservative_verified_return",
                verified_score=row["verified_score"],
                elapsed_seconds=row["elapsed_seconds"],
                metadata=row["metadata"],
            )
            for candidate, row in zip(candidates, rows, strict=True)
        ]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        return [
            CandidateResult(
                candidate=candidate,
                mean_reward=None,
                std_reward=None,
                episode_rewards=[],
                timesteps=config.timesteps,
                seed=config.seed,
                task=config.task,
                status="failed",
                error=error,
                metadata={"sim_backend": config.sim_backend, "candidate_batching": True},
            )
            for candidate in candidates
        ]


def candidates_can_be_batched(
    candidates: list[RewardCandidate],
    config: CandidateEvaluationConfig,
) -> bool:
    return (
        config.timesteps > 0
        and len(candidates) > 1
        and config.sim_backend == "mjwarp"
        and config.mjwarp_evaluator == "ppo"
        and config.mjwarp_rollout_mode == "gpu"
        and config.mjwarp_batch_candidates
    )


def pause_requested(pause_path: Path | None) -> bool:
    return pause_path is not None and pause_path.exists()


def rejected_candidates_to_results(
    rejected: list[tuple[RewardCandidate, str]],
    *,
    task: str,
    seed: int,
    penalty: float,
) -> list[CandidateResult]:
    return [
        CandidateResult(
            candidate=candidate,
            mean_reward=None,
            std_reward=None,
            episode_rewards=[],
            timesteps=0,
            seed=seed + index,
            task=task,
            status="invalid_completion",
            error=error,
            rlvr_reward=penalty,
            rlvr_reward_type="invalid_completion_penalty",
            metadata={"failure_stage": "reward_validation"},
        )
        for index, (candidate, error) in enumerate(rejected)
    ]
