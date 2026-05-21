from __future__ import annotations

import json
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from .adapters import get_adapter
from .generators import initial_population, mutate_candidate
from .hf_generator import HfGeneratorConfig, HfRewardGenerator
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
    status: str = "success"
    error: str | None = None
    elapsed_seconds: float | None = None
    metadata: dict[str, Any] | None = None


def train_and_evaluate(
    candidate: RewardCandidate,
    *,
    task: str,
    timesteps: int,
    eval_episodes: int,
    n_envs: int,
    seed: int,
    device: str,
    sim_backend: str = "sb3",
    worlds_per_candidate: int = 4096,
    mjwarp_evaluator: str = "ppo",
    mjwarp_episode_steps: int = 500,
    mjwarp_policy_iterations: int = 4,
    mjwarp_ppo_horizon: int = 32,
    mjwarp_ppo_epochs: int = 4,
    mjwarp_ppo_minibatch_size: int = 16_384,
    mjwarp_ppo_learning_rate: float = 3.0e-4,
    mjwarp_elite_frac: float = 0.1,
) -> CandidateResult:
    started_at = time.monotonic()
    adapter = get_adapter(task)
    if timesteps <= 0:
        return CandidateResult(
            candidate=candidate,
            mean_reward=None,
            std_reward=None,
            episode_rewards=[],
            timesteps=timesteps,
            seed=seed,
            task=task,
            status="generated_only",
            elapsed_seconds=time.monotonic() - started_at,
        )

    if sim_backend == "mjwarp":
        from .mjwarp_evaluator import MjwarpEvaluatorConfig, train_and_evaluate_mjwarp

        warp_device = "cuda:0" if device in {"auto", "cuda"} else device
        result = train_and_evaluate_mjwarp(
            candidate,
            MjwarpEvaluatorConfig(
                task=task,
                evaluator=mjwarp_evaluator,
                worlds_per_candidate=worlds_per_candidate,
                episode_steps=mjwarp_episode_steps,
                policy_iterations=mjwarp_policy_iterations,
                ppo_horizon=mjwarp_ppo_horizon,
                ppo_epochs=mjwarp_ppo_epochs,
                ppo_minibatch_size=mjwarp_ppo_minibatch_size,
                ppo_learning_rate=mjwarp_ppo_learning_rate,
                elite_frac=mjwarp_elite_frac,
                seed=seed,
                device=warp_device,
                eval_episodes=eval_episodes,
            ),
        )
        return CandidateResult(
            candidate=candidate,
            mean_reward=result["mean_reward"],
            std_reward=result["std_reward"],
            episode_rewards=result["episode_rewards"],
            timesteps=int(result["metadata"]["training_world_steps"]),
            seed=seed,
            task=task,
            elapsed_seconds=result["elapsed_seconds"],
            metadata=result["metadata"],
        )

    if sim_backend != "sb3":
        raise ValueError(f"Unsupported simulation backend: {sim_backend}")

    def make_train_env():
        return RewardWrapper(gym.make(task), candidate.expression, adapter)

    train_env = make_vec_env(make_train_env, n_envs=n_envs, seed=seed)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=6,
        gamma=0.99,
        verbose=0,
        seed=seed,
        device=device,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    train_env.close()

    episode_rewards: list[float] = []
    for episode in range(eval_episodes):
        env = gym.make(task)
        obs, _info = env.reset(seed=seed + 10_000 + episode)
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
        timesteps=timesteps,
        seed=seed,
        task=task,
        elapsed_seconds=time.monotonic() - started_at,
    )


@dataclass(frozen=True)
class RunConfig:
    task: str
    generations: int
    population: int
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


def run_search(
    *,
    task: str,
    generations: int,
    population: int,
    timesteps: int,
    eval_episodes: int,
    n_envs: int,
    seed: int,
    device: str,
    output_dir: Path,
    generator: str = "mock",
    model_id: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    adapter_path: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    load_in_4bit: bool = True,
    sim_backend: str = "sb3",
    worlds_per_candidate: int = 4096,
    mjwarp_evaluator: str = "ppo",
    mjwarp_episode_steps: int = 500,
    mjwarp_policy_iterations: int = 4,
    mjwarp_ppo_horizon: int = 32,
    mjwarp_ppo_epochs: int = 4,
    mjwarp_ppo_minibatch_size: int = 16_384,
    mjwarp_ppo_learning_rate: float = 3.0e-4,
    mjwarp_elite_frac: float = 0.1,
    pause_path: Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> list[CandidateResult]:
    state = load_checkpoint(output_dir) if resume else None
    if state:
        checkpoint_config = state["config"]
        task = checkpoint_config["task"]
        generations = int(checkpoint_config["generations"])
        population = int(checkpoint_config["population"])
        timesteps = int(checkpoint_config["timesteps"])
        eval_episodes = int(checkpoint_config["eval_episodes"])
        n_envs = int(checkpoint_config["n_envs"])
        seed = int(checkpoint_config["seed"])
        device = checkpoint_config["device"]
        generator = checkpoint_config["generator"]
        model_id = checkpoint_config["model_id"]
        adapter_path = checkpoint_config.get("adapter_path")
        max_new_tokens = int(checkpoint_config["max_new_tokens"])
        temperature = float(checkpoint_config["temperature"])
        top_p = float(checkpoint_config["top_p"])
        load_in_4bit = bool(checkpoint_config["load_in_4bit"])
        sim_backend = checkpoint_config.get("sim_backend", "sb3")
        worlds_per_candidate = int(checkpoint_config.get("worlds_per_candidate", 4096))
        mjwarp_evaluator = checkpoint_config.get("mjwarp_evaluator", "ppo")
        mjwarp_episode_steps = int(checkpoint_config.get("mjwarp_episode_steps", 500))
        mjwarp_policy_iterations = int(checkpoint_config.get("mjwarp_policy_iterations", 4))
        mjwarp_ppo_horizon = int(checkpoint_config.get("mjwarp_ppo_horizon", 32))
        mjwarp_ppo_epochs = int(checkpoint_config.get("mjwarp_ppo_epochs", 4))
        mjwarp_ppo_minibatch_size = int(checkpoint_config.get("mjwarp_ppo_minibatch_size", 16_384))
        mjwarp_ppo_learning_rate = float(checkpoint_config.get("mjwarp_ppo_learning_rate", 3.0e-4))
        mjwarp_elite_frac = float(checkpoint_config.get("mjwarp_elite_frac", 0.1))

    run_config = RunConfig(
        task=task,
        generations=generations,
        population=population,
        timesteps=timesteps,
        eval_episodes=eval_episodes,
        n_envs=n_envs,
        seed=seed,
        device=device,
        generator=generator,
        model_id=model_id,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        load_in_4bit=load_in_4bit,
        sim_backend=sim_backend,
        worlds_per_candidate=worlds_per_candidate,
        mjwarp_evaluator=mjwarp_evaluator,
        mjwarp_episode_steps=mjwarp_episode_steps,
        mjwarp_policy_iterations=mjwarp_policy_iterations,
        mjwarp_ppo_horizon=mjwarp_ppo_horizon,
        mjwarp_ppo_epochs=mjwarp_ppo_epochs,
        mjwarp_ppo_minibatch_size=mjwarp_ppo_minibatch_size,
        mjwarp_ppo_learning_rate=mjwarp_ppo_learning_rate,
        mjwarp_elite_frac=mjwarp_elite_frac,
    )
    rng = random.Random(seed)
    prepare_output_dir(output_dir, run_config, resume=resume, overwrite=overwrite)
    get_adapter(task)

    hf_generator = None
    if generator == "hf":
        hf_generator = HfRewardGenerator(
            HfGeneratorConfig(
                model_id=model_id,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                load_in_4bit=load_in_4bit,
            )
        )
        candidates = (
            [candidate_from_dict(row) for row in state["next_candidates"]]
            if state and state.get("next_candidates")
            else hf_generator.generate_population(task=task, population=population, generation=0)
        )
    elif generator == "mock":
        candidates = (
            [candidate_from_dict(row) for row in state["next_candidates"]]
            if state and state.get("next_candidates")
            else initial_population(task, population, rng)
        )
    else:
        raise ValueError(f"Unsupported generator {generator!r}")

    all_results: list[CandidateResult] = (
        [candidate_result_from_dict(row) for row in state["results"]] if state else []
    )
    best_expression: str | None = state.get("best_expression") if state else None
    best_score: float | None = state.get("best_score") if state else None
    start_generation = int(state.get("next_generation", 0)) if state else 0
    completed_keys = {result_key(result.candidate.generation, result.candidate.name) for result in all_results}

    if start_generation >= generations:
        log_event(output_dir, "resume_noop", {"next_generation": start_generation, "generations": generations})
        log_message(output_dir, f"resume noop: next_generation={start_generation} generations={generations}")
        write_results(output_dir, all_results)
        return all_results

    log_event(output_dir, "run_started", {"config": asdict(run_config), "resume": resume})
    log_message(
        output_dir,
        f"run started: task={task} generator={generator} generations={generations} "
        f"population={population} sim_backend={sim_backend}",
    )
    for generation in range(start_generation, generations):
        log_event(output_dir, "generation_started", {"generation": generation})
        log_message(output_dir, f"generation {generation} started")
        if generation > 0 and hf_generator is not None and not (state and generation == start_generation):
            candidates = hf_generator.generate_population(
                task=task,
                population=population,
                generation=generation,
                best_expression=best_expression,
                best_score=best_score,
            )
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
                task=task,
                timesteps=timesteps,
                eval_episodes=eval_episodes,
                n_envs=n_envs,
                seed=seed + generation * 100 + idx,
                device=device,
                sim_backend=sim_backend,
                worlds_per_candidate=worlds_per_candidate,
                mjwarp_evaluator=mjwarp_evaluator,
                mjwarp_episode_steps=mjwarp_episode_steps,
                mjwarp_policy_iterations=mjwarp_policy_iterations,
                mjwarp_ppo_horizon=mjwarp_ppo_horizon,
                mjwarp_ppo_epochs=mjwarp_ppo_epochs,
                mjwarp_ppo_minibatch_size=mjwarp_ppo_minibatch_size,
                mjwarp_ppo_learning_rate=mjwarp_ppo_learning_rate,
                mjwarp_elite_frac=mjwarp_elite_frac,
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
                run_config,
                results=all_results,
                next_generation=generation,
                next_candidates=candidates,
                best_expression=best_expression,
                best_score=best_score,
            )
            if pause_requested(pause_path):
                log_event(output_dir, "run_paused", {"generation": generation, "candidate": candidate.name})
                log_message(output_dir, f"run paused after generation {generation} candidate {candidate.name}")
                return all_results

        generation_results.sort(key=result_sort_key, reverse=True)
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
            },
        )
        log_message(output_dir, f"generation {generation} finished: best={best.name} score={best_score}")
        write_results(output_dir, all_results)

        if hf_generator is None:
            candidates = [best]
            candidates.extend(
                mutate_candidate(best, i, generation + 1, rng) for i in range(max(0, population - 1))
            )
        else:
            candidates = []
        save_checkpoint(
            output_dir,
            run_config,
            results=all_results,
            next_generation=generation + 1,
            next_candidates=candidates,
            best_expression=best_expression,
            best_score=best_score,
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
    *,
    task: str,
    timesteps: int,
    eval_episodes: int,
    n_envs: int,
    seed: int,
    device: str,
    sim_backend: str = "sb3",
    worlds_per_candidate: int = 4096,
    mjwarp_evaluator: str = "ppo",
    mjwarp_episode_steps: int = 500,
    mjwarp_policy_iterations: int = 4,
    mjwarp_ppo_horizon: int = 32,
    mjwarp_ppo_epochs: int = 4,
    mjwarp_ppo_minibatch_size: int = 16_384,
    mjwarp_ppo_learning_rate: float = 3.0e-4,
    mjwarp_elite_frac: float = 0.1,
) -> CandidateResult:
    try:
        return train_and_evaluate(
            candidate,
            task=task,
            timesteps=timesteps,
            eval_episodes=eval_episodes,
            n_envs=n_envs,
            seed=seed,
            device=device,
            sim_backend=sim_backend,
            worlds_per_candidate=worlds_per_candidate,
            mjwarp_evaluator=mjwarp_evaluator,
            mjwarp_episode_steps=mjwarp_episode_steps,
            mjwarp_policy_iterations=mjwarp_policy_iterations,
            mjwarp_ppo_horizon=mjwarp_ppo_horizon,
            mjwarp_ppo_epochs=mjwarp_ppo_epochs,
            mjwarp_ppo_minibatch_size=mjwarp_ppo_minibatch_size,
            mjwarp_ppo_learning_rate=mjwarp_ppo_learning_rate,
            mjwarp_elite_frac=mjwarp_elite_frac,
        )
    except Exception as exc:
        return CandidateResult(
            candidate=candidate,
            mean_reward=None,
            std_reward=None,
            episode_rewards=[],
            timesteps=timesteps,
            seed=seed,
            task=task,
            status="failed",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            metadata={"sim_backend": sim_backend},
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
        "completion": candidate.expression,
        "completion_token_ids": candidate.completion_token_ids,
        "old_logprobs": candidate.old_logprobs,
        "task": result.task,
        "candidate_name": candidate.name,
        "candidate_generation": candidate.generation,
        "train_reward_expression": candidate.expression,
        "verified_reward": result.mean_reward,
        "verified_reward_std": result.std_reward,
        "verified_reward_episodes": result.episode_rewards,
        "verified_reward_type": result.verified_reward_type,
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
) -> None:
    payload = {
        "config": asdict(config),
        "next_generation": next_generation,
        "next_candidates": [asdict(candidate) for candidate in next_candidates],
        "best_expression": best_expression,
        "best_score": best_score,
        "results": [candidate_result_to_dict(result) for result in results],
        "updated_at": time.time(),
    }
    atomic_write_text(output_dir / "checkpoint.json", json.dumps(payload, indent=2))


def load_checkpoint(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "checkpoint.json").read_text(encoding="utf-8"))


def append_rlvr_record(output_dir: Path, result: CandidateResult) -> None:
    with (output_dir / "rlvr_records.incremental.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_rlvr_record(result), sort_keys=True) + "\n")


def log_event(output_dir: Path, event: str, payload: dict[str, Any]) -> None:
    row = {"time": time.time(), "event": event, **payload}
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def log_message(output_dir: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (output_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    atomic_write_text(path, "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""))


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


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
        generation=int(row.get("generation", 0)),
        generator_type=row.get("generator_type", "mock"),
        generator_checkpoint=row.get("generator_checkpoint", "mock-v1"),
        completion_token_ids=row.get("completion_token_ids"),
        old_logprobs=row.get("old_logprobs"),
    )


def pause_requested(pause_path: Path | None) -> bool:
    return pause_path is not None and pause_path.exists()
