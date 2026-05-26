from __future__ import annotations

import json
import random
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import get_adapter
from .generators import initial_population, mutate_candidate
from .hf_generator import HfGeneratorConfig, HfRewardGenerator
from .io_utils import atomic_write_text, write_jsonl
from .rewards import RewardCandidate, RewardWrapper


@dataclass(frozen=True)
class CandidateResult:
    candidate: RewardCandidate
    mean_reward: float | None
    std_reward: float | None
    episode_rewards: list[float]
    timesteps: int
    seed: int
    task: str
    verified_reward_type: str = "true_env_return"
    rlvr_reward: float | None = None
    rlvr_reward_type: str | None = None
    status: str = "success"
    error: str | None = None
    elapsed_seconds: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidateEvaluationConfig:
    task: str
    timesteps: int
    eval_episodes: int
    n_envs: int
    seed: int
    device: str
    sim_backend: str = "sb3"
    worlds_per_candidate: int = 4096
    mjwarp_evaluator: str = "ppo"
    mjwarp_episode_steps: int = 500
    mjwarp_policy_iterations: int = 4
    mjwarp_ppo_horizon: int = 32
    mjwarp_ppo_epochs: int = 4
    mjwarp_ppo_minibatch_size: int = 16_384
    mjwarp_ppo_learning_rate: float = 3.0e-4
    mjwarp_elite_frac: float = 0.1


@dataclass(frozen=True)
class RunConfig:
    task: str
    generations: int
    population: int
    eureka_elites: int
    timesteps: int
    eval_episodes: int
    n_envs: int
    seed: int
    device: str
    generator: str
    model_id: str
    adapter_path: str | None
    max_new_tokens: int
    temperature: float
    top_p: float
    load_in_4bit: bool
    sim_backend: str = "sb3"
    worlds_per_candidate: int = 4096
    mjwarp_evaluator: str = "ppo"
    mjwarp_episode_steps: int = 500
    mjwarp_policy_iterations: int = 4
    mjwarp_ppo_horizon: int = 32
    mjwarp_ppo_epochs: int = 4
    mjwarp_ppo_minibatch_size: int = 16_384
    mjwarp_ppo_learning_rate: float = 3.0e-4
    mjwarp_elite_frac: float = 0.1
    include_negative_rlvr_samples: bool = True
    negative_rlvr_margin: float = 1.0

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RunConfig":
        return cls(
            task=row["task"],
            generations=int(row["generations"]),
            population=int(row["population"]),
            eureka_elites=int(row.get("eureka_elites", 4)),
            timesteps=int(row["timesteps"]),
            eval_episodes=int(row["eval_episodes"]),
            n_envs=int(row["n_envs"]),
            seed=int(row["seed"]),
            device=row["device"],
            generator=row["generator"],
            model_id=row["model_id"],
            adapter_path=row.get("adapter_path"),
            max_new_tokens=int(row["max_new_tokens"]),
            temperature=float(row["temperature"]),
            top_p=float(row["top_p"]),
            load_in_4bit=bool(row["load_in_4bit"]),
            sim_backend=row.get("sim_backend", "sb3"),
            worlds_per_candidate=int(row.get("worlds_per_candidate", 4096)),
            mjwarp_evaluator=row.get("mjwarp_evaluator", "ppo"),
            mjwarp_episode_steps=int(row.get("mjwarp_episode_steps", 500)),
            mjwarp_policy_iterations=int(row.get("mjwarp_policy_iterations", 4)),
            mjwarp_ppo_horizon=int(row.get("mjwarp_ppo_horizon", 32)),
            mjwarp_ppo_epochs=int(row.get("mjwarp_ppo_epochs", 4)),
            mjwarp_ppo_minibatch_size=int(row.get("mjwarp_ppo_minibatch_size", 16_384)),
            mjwarp_ppo_learning_rate=float(row.get("mjwarp_ppo_learning_rate", 3.0e-4)),
            mjwarp_elite_frac=float(row.get("mjwarp_elite_frac", 0.1)),
            include_negative_rlvr_samples=bool(row.get("include_negative_rlvr_samples", True)),
            negative_rlvr_margin=float(row.get("negative_rlvr_margin", 1.0)),
        )

    def evaluation_config(self, seed: int) -> CandidateEvaluationConfig:
        return CandidateEvaluationConfig(
            task=self.task,
            timesteps=self.timesteps,
            eval_episodes=self.eval_episodes,
            n_envs=self.n_envs,
            seed=seed,
            device=self.device,
            sim_backend=self.sim_backend,
            worlds_per_candidate=self.worlds_per_candidate,
            mjwarp_evaluator=self.mjwarp_evaluator,
            mjwarp_episode_steps=self.mjwarp_episode_steps,
            mjwarp_policy_iterations=self.mjwarp_policy_iterations,
            mjwarp_ppo_horizon=self.mjwarp_ppo_horizon,
            mjwarp_ppo_epochs=self.mjwarp_ppo_epochs,
            mjwarp_ppo_minibatch_size=self.mjwarp_ppo_minibatch_size,
            mjwarp_ppo_learning_rate=self.mjwarp_ppo_learning_rate,
            mjwarp_elite_frac=self.mjwarp_elite_frac,
        )


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
        from .mjwarp_evaluator import MjwarpEvaluatorConfig, train_and_evaluate_mjwarp

        warp_device = "cuda:0" if config.device in {"auto", "cuda"} else config.device
        result = train_and_evaluate_mjwarp(
            candidate,
            MjwarpEvaluatorConfig(
                task=config.task,
                evaluator=config.mjwarp_evaluator,
                worlds_per_candidate=config.worlds_per_candidate,
                episode_steps=config.mjwarp_episode_steps,
                policy_iterations=config.mjwarp_policy_iterations,
                ppo_horizon=config.mjwarp_ppo_horizon,
                ppo_epochs=config.mjwarp_ppo_epochs,
                ppo_minibatch_size=config.mjwarp_ppo_minibatch_size,
                ppo_learning_rate=config.mjwarp_ppo_learning_rate,
                elite_frac=config.mjwarp_elite_frac,
                seed=config.seed,
                device=warp_device,
                eval_episodes=config.eval_episodes,
            ),
        )
        return CandidateResult(
            candidate=candidate,
            mean_reward=result["mean_reward"],
            std_reward=result["std_reward"],
            episode_rewards=result["episode_rewards"],
            timesteps=int(result["metadata"]["training_world_steps"]),
            seed=config.seed,
            task=config.task,
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

    return CandidateResult(
        candidate=candidate,
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        episode_rewards=episode_rewards,
        timesteps=config.timesteps,
        seed=config.seed,
        task=config.task,
        elapsed_seconds=time.monotonic() - started_at,
    )


def run_search(
    config: RunConfig,
    *,
    output_dir: Path,
    pause_path: Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> list[CandidateResult]:
    state = load_checkpoint(output_dir) if resume else None
    if state:
        config = RunConfig.from_dict(state["config"])

    rng = random.Random(config.seed)
    prepare_output_dir(output_dir, config, resume=resume, overwrite=overwrite)
    get_adapter(config.task)

    hf_generator = None
    if config.generator == "hf":
        hf_generator = HfRewardGenerator(
            HfGeneratorConfig(
                model_id=config.model_id,
                adapter_path=config.adapter_path,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                load_in_4bit=config.load_in_4bit,
            )
        )
        candidates = (
            [candidate_from_dict(row) for row in state["next_candidates"]]
            if state and state.get("next_candidates")
            else hf_generator.generate_population(task=config.task, population=config.population, generation=0)
        )
        pending_rejections = [] if state else hf_generator.drain_rejected_candidates()
    elif config.generator == "mock":
        candidates = (
            [candidate_from_dict(row) for row in state["next_candidates"]]
            if state and state.get("next_candidates")
            else initial_population(config.task, config.population, rng)
        )
        pending_rejections = []
    else:
        raise ValueError(f"Unsupported generator {config.generator!r}")

    all_results: list[CandidateResult] = (
        [candidate_result_from_dict(row) for row in state["results"]] if state else []
    )
    best_expression: str | None = state.get("best_expression") if state else None
    best_score: float | None = state.get("best_score") if state else None
    elite_context: list[dict[str, Any]] = list(state.get("elite_context", [])) if state else []
    evolution_feedback: str | None = state.get("evolution_feedback") if state else None
    start_generation = int(state.get("next_generation", 0)) if state else 0
    completed_keys = {result_key(result.candidate.generation, result.candidate.name) for result in all_results}

    if start_generation >= config.generations:
        log_event(output_dir, "resume_noop", {"next_generation": start_generation, "generations": config.generations})
        log_message(output_dir, f"resume noop: next_generation={start_generation} generations={config.generations}")
        write_results(output_dir, all_results)
        return all_results

    log_event(output_dir, "run_started", {"config": asdict(config), "resume": resume})
    log_message(
        output_dir,
        f"run started: task={config.task} generator={config.generator} generations={config.generations} "
        f"population={config.population} sim_backend={config.sim_backend}",
    )
    for generation in range(start_generation, config.generations):
        log_event(output_dir, "generation_started", {"generation": generation})
        log_message(output_dir, f"generation {generation} started")
        if generation > 0 and hf_generator is not None and not (state and generation == start_generation):
            candidates = hf_generator.generate_population(
                task=config.task,
                population=config.population,
                generation=generation,
                best_expression=best_expression,
                best_score=best_score,
                elites=elite_context,
                evolution_feedback=evolution_feedback,
            )
            pending_rejections = hf_generator.drain_rejected_candidates()
        generation_results = []
        for idx, candidate in enumerate(candidates):
            key = result_key(generation, candidate.name)
            if key in completed_keys:
                existing = next(result for result in all_results if result_key(result.candidate.generation, result.candidate.name) == key)
                generation_results.append(existing)
                log_event(output_dir, "candidate_skipped", {"generation": generation, "candidate": candidate.name})
                log_message(output_dir, f"generation {generation} candidate {candidate.name} skipped")
                continue
            log_event(output_dir, "candidate_started", {"generation": generation, "candidate": candidate.name})
            log_message(output_dir, f"generation {generation} candidate {candidate.name} started")
            result = run_candidate_safely(
                candidate,
                config.evaluation_config(config.seed + generation * 100 + idx),
            )
            generation_results.append(result)
            all_results.append(result)
            completed_keys.add(key)
            append_rlvr_record(output_dir, result)
            log_event(
                output_dir,
                "candidate_finished",
                {
                    "generation": generation,
                    "candidate": candidate.name,
                    "status": result.status,
                    "mean_reward": result.mean_reward,
                    "elapsed_seconds": result.elapsed_seconds,
                },
            )
            log_message(
                output_dir,
                f"generation {generation} candidate {candidate.name} finished: "
                f"status={result.status} mean_reward={result.mean_reward} elapsed_seconds={result.elapsed_seconds}",
            )
            write_results(output_dir, all_results)
            save_checkpoint(
                output_dir,
                config,
                results=all_results,
                next_generation=generation,
                next_candidates=candidates,
                best_expression=best_expression,
                best_score=best_score,
                elite_context=elite_context,
                evolution_feedback=evolution_feedback,
            )
            if pause_requested(pause_path):
                log_event(output_dir, "run_paused", {"generation": generation, "candidate": candidate.name})
                log_message(output_dir, f"run paused after generation {generation} candidate {candidate.name}")
                return all_results

        if config.include_negative_rlvr_samples:
            failure_penalty = negative_reward_for_generation(generation_results, config.negative_rlvr_margin)
            generation_results = assign_failed_evaluation_penalties(generation_results, failure_penalty)
            all_results = replace_generation_results(all_results, generation_results)
            rejected_results = rejected_candidates_to_results(
                pending_rejections,
                task=config.task,
                seed=config.seed + generation * 100,
                penalty=failure_penalty,
            )
            for result in rejected_results:
                all_results.append(result)
                append_rlvr_record(output_dir, result)
                log_event(
                    output_dir,
                    "candidate_rejected",
                    {
                        "generation": generation,
                        "candidate": result.candidate.name,
                        "status": result.status,
                        "rlvr_reward": result.rlvr_reward,
                        "error": result.error,
                    },
                )
        pending_rejections = []
        if not generation_results:
            write_results(output_dir, all_results)
            save_checkpoint(
                output_dir,
                config,
                results=all_results,
                next_generation=config.generations,
                next_candidates=[],
                best_expression=best_expression,
                best_score=best_score,
                elite_context=elite_context,
                evolution_feedback=evolution_feedback,
            )
            log_event(
                output_dir,
                "generation_no_executable_candidates",
                {"generation": generation, "negative_records": len(all_results)},
            )
            log_message(output_dir, f"generation {generation} produced no executable reward candidates")
            return all_results

        generation_results.sort(key=result_sort_key, reverse=True)
        generation_results = annotate_generation_results(generation_results, config.eureka_elites)
        all_results = replace_generation_results(all_results, generation_results)
        elite_context = elite_context_from_results(generation_results, config.eureka_elites)
        evolution_feedback = format_generation_feedback(generation_results, config.eureka_elites)
        best = generation_results[0].candidate
        best_expression = best.expression
        best_score = generation_results[0].mean_reward
        log_event(
            output_dir,
            "generation_finished",
            {
                "generation": generation,
                "best_candidate": best.name,
                "best_score": best_score,
                "elite_context": elite_context,
                "evolution_feedback": evolution_feedback,
            },
        )
        log_message(output_dir, f"generation {generation} finished: best={best.name} score={best_score}")
        write_results(output_dir, all_results)

        if hf_generator is None:
            parents = [result.candidate for result in generation_results[: max(1, min(config.eureka_elites, len(generation_results)))]]
            parent_scores = {result.candidate.name: result.mean_reward for result in generation_results}
            candidates = []
            for i in range(config.population):
                parent = parents[i % len(parents)]
                child = mutate_candidate(parent, i, generation + 1, rng)
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
        else:
            candidates = []
        save_checkpoint(
            output_dir,
            config,
            results=all_results,
            next_generation=generation + 1,
            next_candidates=candidates,
            best_expression=best_expression,
            best_score=best_score,
            elite_context=elite_context,
            evolution_feedback=evolution_feedback,
        )
        state = None
        if pause_requested(pause_path):
            log_event(output_dir, "run_paused", {"generation": generation})
            log_message(output_dir, f"run paused after generation {generation}")
            return all_results

    write_results(output_dir, all_results)
    log_event(output_dir, "run_finished", {"results_count": len(all_results), "best_score": best_score})
    log_message(output_dir, f"run finished: results_count={len(all_results)} best_score={best_score}")
    return all_results


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


def write_results(output_dir: Path, results: list[CandidateResult]) -> None:
    payload = []
    for result in sorted(results, key=result_sort_key, reverse=True):
        row = asdict(result)
        row["candidate"] = asdict(result.candidate)
        payload.append(row)

    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "rlvr_records.jsonl", [to_rlvr_record(result) for result in results])
    if payload:
        best = payload[0]
        (output_dir / "best_reward.py").write_text(
            "# Best discovered reward expression\n"
            f"REWARD_EXPRESSION = {best['candidate']['expression']!r}\n"
            f"MEAN_REWARD = {best['mean_reward']!r}\n",
            encoding="utf-8",
        )


def result_sort_key(result: CandidateResult) -> float:
    if result.mean_reward is None:
        return float("-inf")
    return result.mean_reward


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
) -> None:
    payload = {
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
    atomic_write_text(output_dir / "checkpoint.json", json.dumps(payload, indent=2))


def load_checkpoint(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "checkpoint.json").read_text(encoding="utf-8"))


def append_rlvr_record(output_dir: Path, result: CandidateResult) -> None:
    write_jsonl(output_dir / "rlvr_records.incremental.jsonl", [to_rlvr_record(result)], append=True)


def log_event(output_dir: Path, event: str, payload: dict[str, Any]) -> None:
    row = {"time": time.time(), "event": event, **payload}
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def log_message(output_dir: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (output_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def result_key(generation: int, candidate_name: str) -> str:
    return f"{generation}:{candidate_name}"


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


def pause_requested(pause_path: Path | None) -> bool:
    return pause_path is not None and pause_path.exists()


def elite_context_from_results(results: list[CandidateResult], elite_count: int) -> list[dict[str, Any]]:
    if elite_count < 1:
        raise ValueError("--eureka-elites must be at least 1")
    elites = sorted(results, key=result_sort_key, reverse=True)[:elite_count]
    return [
        {
            "rank": rank,
            "name": result.candidate.name,
            "expression": result.candidate.expression,
            "score": result.mean_reward,
            "status": result.status,
            "verified_reward_type": result.verified_reward_type,
        }
        for rank, result in enumerate(elites, start=1)
    ]


def annotate_generation_results(results: list[CandidateResult], elite_count: int) -> list[CandidateResult]:
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


def replace_generation_results(
    all_results: list[CandidateResult], generation_results: list[CandidateResult]
) -> list[CandidateResult]:
    updates = {
        result_key(result.candidate.generation, result.candidate.name): result for result in generation_results
    }
    return [
        updates.get(result_key(result.candidate.generation, result.candidate.name), result)
        for result in all_results
    ]


def negative_reward_for_generation(results: list[CandidateResult], margin: float) -> float:
    if margin <= 0:
        raise ValueError("--negative-rlvr-margin must be greater than 0")
    successful_scores = [
        result.mean_reward
        for result in results
        if result.status == "success" and result.mean_reward is not None
    ]
    baseline = min(successful_scores) if successful_scores else 0.0
    return float(baseline) - margin


def assign_failed_evaluation_penalties(
    results: list[CandidateResult], penalty: float
) -> list[CandidateResult]:
    return [
        replace(
            result,
            rlvr_reward=penalty,
            rlvr_reward_type="failed_evaluation_penalty",
        )
        if result.status == "failed"
        else result
        for result in results
    ]


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


def format_generation_feedback(results: list[CandidateResult], elite_count: int) -> str:
    ranked = sorted(results, key=result_sort_key, reverse=True)
    elites = ranked[:elite_count]
    rejected = ranked[elite_count:]
    successful_rewards = [result.mean_reward for result in ranked if result.mean_reward is not None]
    lines = ["Ranked by verified true environment return."]
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


def format_result_feedback_row(rank: int, result: CandidateResult, *, selected: bool) -> list[str]:
    score = "n/a" if result.mean_reward is None else f"{result.mean_reward:.4f}"
    prefix = "ELITE" if selected else "NON_ELITE"
    lines = [
        f"{rank}. {prefix} name={result.candidate.name} status={result.status} verified_return={score}",
        f"   expression={result.candidate.expression}",
    ]
    metadata = result.metadata or {}
    best_shaped = metadata.get("best_shaped_return")
    summaries = metadata.get("iteration_summaries") or []
    best_internal_true = None
    if summaries:
        true_values = [
            item.get("best_true_return_in_population")
            for item in summaries
            if item.get("best_true_return_in_population") is not None
        ]
        if true_values:
            best_internal_true = max(float(value) for value in true_values)
    diagnostics = []
    if best_shaped is not None:
        diagnostics.append(f"best_shaped_return={float(best_shaped):.4f}")
    if best_internal_true is not None:
        diagnostics.append(f"best_internal_true_return={best_internal_true:.4f}")
    if result.error:
        diagnostics.append(f"error={result.error.splitlines()[0]}")
    if diagnostics:
        lines.append("   diagnostics=" + ", ".join(diagnostics))
    component_stats = latest_component_stats(metadata)
    if component_stats:
        parts = []
        for name, stats in component_stats.items():
            parts.append(
                f"{name}:mean={float(stats['mean']):.4f},min={float(stats['min']):.4f},max={float(stats['max']):.4f}"
            )
        lines.append("   reward_component_stats=" + "; ".join(parts))
    return lines


def latest_component_stats(metadata: dict[str, Any]) -> dict[str, dict[str, float]]:
    summaries = metadata.get("iteration_summaries") or []
    for item in reversed(summaries):
        stats = item.get("reward_component_stats")
        if stats:
            return stats
    return {}
