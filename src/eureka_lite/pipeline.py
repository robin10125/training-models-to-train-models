from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .adapters import ANT_TASK
from .cli_options import add_eureka_options, add_generation_options, add_mjwarp_options, add_negative_sample_options
from .hf_generator import DEFAULT_HF_MODEL_ID
from .rlvr_trainer import RlvrTrainerConfig, train_rlvr
from .search import RunConfig, run_search


@dataclass(frozen=True)
class FullPipelineConfig:
    task: str
    model_id: str
    run_root: str
    iterations: int
    population: int
    generations: int
    eureka_elites: int
    worlds_per_candidate: int
    mjwarp_evaluator: str
    mjwarp_episode_steps: int
    mjwarp_policy_iterations: int
    mjwarp_ppo_horizon: int
    mjwarp_ppo_epochs: int
    mjwarp_ppo_minibatch_size: int
    mjwarp_ppo_learning_rate: float
    mjwarp_elite_frac: float
    mjwarp_rollout_mode: str
    mjwarp_verified_evaluator: str
    mjwarp_verification_steps: int
    mjwarp_batch_candidates: bool
    mjwarp_cuda_graph: bool
    include_negative_rlvr_samples: bool
    negative_rlvr_margin: float
    eval_episodes: int
    seed: int
    device: str
    max_new_tokens: int
    temperature: float
    top_p: float
    load_in_4bit: bool
    trainer_algorithm: str
    trainer_epochs: int
    trainer_batch_size: int
    trainer_learning_rate: float
    trainer_max_length: int
    trainer_max_grad_norm: float
    trainer_lora_r: int
    trainer_lora_alpha: int
    trainer_lora_dropout: float
    trainer_clip_epsilon: float
    trainer_beta_kl: float
    overwrite_collection: bool
    force_train: bool
    mjwarp_training_episode_horizon: int = 1000
    mjwarp_ppo_init_mode: str = "base"
    mjwarp_base_policy_checkpoint: str | None = "checkpoints/ant_mjwarp_warm_start_1500.pt"
    mjwarp_verified_audit_gym: bool = False
    mjwarp_verified_audit_max_abs_diff: float | None = None
    mjwarp_reward_backend: str = "eager"
    checkpoint_retention: str = "all"


def run_full_pipeline(config: FullPipelineConfig) -> dict[str, Any]:
    started_at = time.monotonic()
    run_root = Path(config.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    pipeline_state_path = run_root / "pipeline_state.json"
    pause_path = run_root / "PAUSE"

    adapter_path: str | None = None
    iteration_summaries = []
    for iteration in range(config.iterations):
        if pause_path.exists():
            summary = build_pipeline_summary(config, iteration_summaries, started_at, status="paused")
            pipeline_state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
        collection_dir = run_root / f"iteration_{iteration:03d}" / "collection"
        adapter_dir = run_root / f"iteration_{iteration:03d}" / "adapter"
        collection_resume = (collection_dir / "checkpoint.json").exists() and not config.overwrite_collection
        collection_started = time.monotonic()
        collection_memory_before = cuda_memory_snapshot(config.device)
        results = run_search(
            collection_config(config, iteration, adapter_path),
            output_dir=collection_dir,
            pause_path=pause_path,
            resume=collection_resume,
            overwrite=config.overwrite_collection,
        )
        collection_seconds = time.monotonic() - collection_started
        collection_memory_after = cuda_memory_snapshot(config.device)

        records_path = collection_dir / "rlvr_records.jsonl"
        best_verified_reward = max(
            (result.mean_reward for result in results if result.mean_reward is not None),
            default=None,
        )
        if pause_path.exists():
            iteration_summaries.append(
                iteration_summary(
                    iteration=iteration,
                    collection_dir=collection_dir,
                    adapter_dir=adapter_dir,
                    records_path=records_path,
                    generator_adapter_path=adapter_path,
                    results_count=len(results),
                    best_verified_reward=best_verified_reward,
                    trainer_status="not_started_pause_requested",
                    trainer_final_loss=None,
                    collection_seconds=collection_seconds,
                    training_seconds=None,
                    collection_memory_before=collection_memory_before,
                    collection_memory_after=collection_memory_after,
                    training_memory_before=None,
                    training_memory_after=None,
                )
            )
            summary = build_pipeline_summary(config, iteration_summaries, started_at, status="paused")
            pipeline_state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
        if not records_path.exists() or records_path.stat().st_size == 0:
            raise FileNotFoundError(f"Collection did not produce RLVR records at {records_path}")

        metrics_path = adapter_dir / "trainer_metrics.json"
        training_started = time.monotonic()
        training_memory_before = cuda_memory_snapshot(config.device)
        if metrics_path.exists() and not config.force_train:
            trainer_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            trainer_status = "skipped_existing_adapter"
        else:
            trainer_metrics = train_rlvr(trainer_config(config, records_path, adapter_dir, adapter_path, pause_path))
            trainer_status = trainer_metrics.get("status", "trained")
        training_seconds = time.monotonic() - training_started
        training_memory_after = cuda_memory_snapshot(config.device)

        iteration_summaries.append(
            iteration_summary(
                iteration=iteration,
                collection_dir=collection_dir,
                adapter_dir=adapter_dir,
                records_path=records_path,
                generator_adapter_path=adapter_path,
                results_count=len(results),
                best_verified_reward=best_verified_reward,
                trainer_status=trainer_status,
                trainer_final_loss=trainer_metrics.get("final_loss"),
                collection_seconds=collection_seconds,
                training_seconds=training_seconds,
                collection_memory_before=collection_memory_before,
                collection_memory_after=collection_memory_after,
                training_memory_before=training_memory_before,
                training_memory_after=training_memory_after,
            )
        )
        if trainer_status == "paused":
            summary = build_pipeline_summary(config, iteration_summaries, started_at, status="paused")
            pipeline_state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
        adapter_path = adapter_dir.as_posix()
        summary = build_pipeline_summary(config, iteration_summaries, started_at, status="running")
        pipeline_state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary = build_pipeline_summary(config, iteration_summaries, started_at, status="completed")
    pipeline_state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def collection_config(config: FullPipelineConfig, iteration: int, adapter_path: str | None) -> RunConfig:
    return RunConfig(
        task=config.task,
        generations=config.generations,
        population=config.population,
        timesteps=1,
        eval_episodes=config.eval_episodes,
        n_envs=1,
        seed=config.seed + iteration * 10_000,
        device=config.device,
        eureka_elites=config.eureka_elites,
        generator="hf",
        model_id=config.model_id,
        adapter_path=adapter_path,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        load_in_4bit=config.load_in_4bit,
        sim_backend="mjwarp",
        worlds_per_candidate=config.worlds_per_candidate,
        mjwarp_evaluator=config.mjwarp_evaluator,
        mjwarp_episode_steps=config.mjwarp_episode_steps,
        mjwarp_training_episode_horizon=config.mjwarp_training_episode_horizon,
        mjwarp_policy_iterations=config.mjwarp_policy_iterations,
        mjwarp_ppo_horizon=config.mjwarp_ppo_horizon,
        mjwarp_ppo_epochs=config.mjwarp_ppo_epochs,
        mjwarp_ppo_minibatch_size=config.mjwarp_ppo_minibatch_size,
        mjwarp_ppo_learning_rate=config.mjwarp_ppo_learning_rate,
        mjwarp_ppo_init_mode=config.mjwarp_ppo_init_mode,
        mjwarp_base_policy_checkpoint=config.mjwarp_base_policy_checkpoint,
        mjwarp_elite_frac=config.mjwarp_elite_frac,
        mjwarp_rollout_mode=config.mjwarp_rollout_mode,
        mjwarp_verified_evaluator=config.mjwarp_verified_evaluator,
        mjwarp_verification_steps=config.mjwarp_verification_steps,
        mjwarp_verified_audit_gym=config.mjwarp_verified_audit_gym,
        mjwarp_verified_audit_max_abs_diff=config.mjwarp_verified_audit_max_abs_diff,
        mjwarp_reward_backend=config.mjwarp_reward_backend,
        mjwarp_batch_candidates=config.mjwarp_batch_candidates,
        mjwarp_cuda_graph=config.mjwarp_cuda_graph,
        include_negative_rlvr_samples=config.include_negative_rlvr_samples,
        negative_rlvr_margin=config.negative_rlvr_margin,
    )


def trainer_config(
    config: FullPipelineConfig,
    records_path: Path,
    adapter_dir: Path,
    adapter_path: str | None,
    pause_path: Path,
) -> RlvrTrainerConfig:
    return RlvrTrainerConfig(
        records_path=str(records_path),
        output_dir=str(adapter_dir),
        model_id=config.model_id,
        adapter_path=adapter_path,
        max_length=config.trainer_max_length,
        epochs=config.trainer_epochs,
        batch_size=config.trainer_batch_size,
        learning_rate=config.trainer_learning_rate,
        max_grad_norm=config.trainer_max_grad_norm,
        lora_r=config.trainer_lora_r,
        lora_alpha=config.trainer_lora_alpha,
        lora_dropout=config.trainer_lora_dropout,
        load_in_4bit=config.load_in_4bit,
        algorithm=config.trainer_algorithm,
        clip_epsilon=config.trainer_clip_epsilon,
        beta_kl=config.trainer_beta_kl,
        resume=True,
        pause_path=pause_path.as_posix(),
        checkpoint_retention=config.checkpoint_retention,
    )


def build_pipeline_summary(
    config: FullPipelineConfig,
    iteration_summaries: list[dict[str, Any]],
    started_at: float,
    *,
    status: str,
) -> dict[str, Any]:
    summary = {
        "config": asdict(config),
        "run_root": config.run_root,
        "status": status,
        "iterations_completed": len(iteration_summaries),
        "iterations": iteration_summaries,
        "latest_adapter_output_dir": iteration_summaries[-1]["adapter_output_dir"] if iteration_summaries else None,
        "best_verified_reward": max(
            (
                item["best_verified_reward"]
                for item in iteration_summaries
                if item["best_verified_reward"] is not None
            ),
            default=None,
        ),
        "elapsed_seconds": time.monotonic() - started_at,
        "updated_at": time.time(),
    }
    return summary


def iteration_summary(
    *,
    iteration: int,
    collection_dir: Path,
    adapter_dir: Path,
    records_path: Path,
    generator_adapter_path: str | None,
    results_count: int,
    best_verified_reward: float | None,
    trainer_status: str,
    trainer_final_loss: float | None,
    collection_seconds: float | None = None,
    training_seconds: float | None = None,
    collection_memory_before: dict[str, Any] | None = None,
    collection_memory_after: dict[str, Any] | None = None,
    training_memory_before: dict[str, Any] | None = None,
    training_memory_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "collection_output_dir": collection_dir.as_posix(),
        "adapter_output_dir": adapter_dir.as_posix(),
        "records_path": records_path.as_posix(),
        "generator_adapter_path": generator_adapter_path,
        "collection_results": results_count,
        "best_verified_reward": best_verified_reward,
        "trainer_status": trainer_status,
        "trainer_final_loss": trainer_final_loss,
        "phase_telemetry": {
            "collection_seconds": collection_seconds,
            "training_seconds": training_seconds,
            "collection_memory_before": collection_memory_before,
            "collection_memory_after": collection_memory_after,
            "training_memory_before": training_memory_before,
            "training_memory_after": training_memory_after,
        },
    }


def cuda_memory_snapshot(device: str) -> dict[str, Any] | None:
    if device not in {"cuda", "auto"} and not device.startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        target = torch.device("cuda:0" if device in {"cuda", "auto"} else device)
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(target)),
            "reserved_bytes": int(torch.cuda.memory_reserved(target)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(target)),
        }
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full MJWarp EUREKA collection + RLVR training pipeline.")
    parser.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--run-root", type=Path, default=Path("runs/deepseek_lite_ant_mjwarp_rlvr"))
    parser.add_argument("--iterations", type=int, default=3)
    add_eureka_options(parser, population_default=16)
    add_mjwarp_options(parser)
    add_negative_sample_options(parser)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    add_generation_options(parser)
    parser.add_argument("--trainer-algorithm", choices=["weighted_sft", "grpo"], default="grpo")
    parser.add_argument("--trainer-epochs", type=int, default=1)
    parser.add_argument("--trainer-batch-size", type=int, default=1)
    parser.add_argument("--trainer-learning-rate", type=float, default=5e-5)
    parser.add_argument("--trainer-max-length", type=int, default=8192)
    parser.add_argument("--trainer-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--trainer-lora-r", type=int, default=16)
    parser.add_argument("--trainer-lora-alpha", type=int, default=32)
    parser.add_argument("--trainer-lora-dropout", type=float, default=0.05)
    parser.add_argument("--trainer-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--trainer-beta-kl", type=float, default=0.01)
    parser.add_argument("--overwrite-collection", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--checkpoint-retention",
        choices=["all", "latest"],
        default="all",
        help="Retain all RLVR epoch checkpoints or only the newest resumable checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if args.population < 2 and args.trainer_algorithm == "grpo":
        raise SystemExit("--trainer-algorithm grpo requires --population at least 2")
    if args.eureka_elites < 1:
        raise SystemExit("--eureka-elites must be at least 1")
    if args.negative_rlvr_margin <= 0:
        raise SystemExit("--negative-rlvr-margin must be greater than 0")
    config = FullPipelineConfig(
        task=ANT_TASK,
        model_id=args.model_id,
        run_root=args.run_root.as_posix(),
        iterations=args.iterations,
        population=args.population,
        generations=args.generations,
        eureka_elites=args.eureka_elites,
        worlds_per_candidate=args.worlds_per_candidate,
        mjwarp_evaluator=args.mjwarp_evaluator,
        mjwarp_episode_steps=args.mjwarp_episode_steps,
        mjwarp_training_episode_horizon=args.mjwarp_training_episode_horizon,
        mjwarp_policy_iterations=args.mjwarp_policy_iterations,
        mjwarp_ppo_horizon=args.mjwarp_ppo_horizon,
        mjwarp_ppo_epochs=args.mjwarp_ppo_epochs,
        mjwarp_ppo_minibatch_size=args.mjwarp_ppo_minibatch_size,
        mjwarp_ppo_learning_rate=args.mjwarp_ppo_learning_rate,
        mjwarp_ppo_init_mode=args.mjwarp_ppo_init_mode,
        mjwarp_base_policy_checkpoint=args.mjwarp_base_policy_checkpoint,
        mjwarp_elite_frac=args.mjwarp_elite_frac,
        mjwarp_rollout_mode=args.mjwarp_rollout_mode,
        mjwarp_verified_evaluator=args.mjwarp_verified_evaluator,
        mjwarp_verification_steps=args.mjwarp_verification_steps,
        mjwarp_verified_audit_gym=args.mjwarp_verified_audit_gym,
        mjwarp_verified_audit_max_abs_diff=args.mjwarp_verified_audit_max_abs_diff,
        mjwarp_reward_backend=args.mjwarp_reward_backend,
        mjwarp_batch_candidates=not args.no_mjwarp_candidate_batching,
        mjwarp_cuda_graph=not args.no_mjwarp_cuda_graph,
        include_negative_rlvr_samples=not args.no_negative_rlvr_samples,
        negative_rlvr_margin=args.negative_rlvr_margin,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=not args.no_4bit,
        trainer_algorithm=args.trainer_algorithm,
        trainer_epochs=args.trainer_epochs,
        trainer_batch_size=args.trainer_batch_size,
        trainer_learning_rate=args.trainer_learning_rate,
        trainer_max_length=args.trainer_max_length,
        trainer_max_grad_norm=args.trainer_max_grad_norm,
        trainer_lora_r=args.trainer_lora_r,
        trainer_lora_alpha=args.trainer_lora_alpha,
        trainer_lora_dropout=args.trainer_lora_dropout,
        trainer_clip_epsilon=args.trainer_clip_epsilon,
        trainer_beta_kl=args.trainer_beta_kl,
        overwrite_collection=args.overwrite_collection,
        force_train=args.force_train,
        checkpoint_retention=args.checkpoint_retention,
    )
    summary = run_full_pipeline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
