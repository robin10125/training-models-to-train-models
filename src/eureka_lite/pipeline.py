from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hf_generator import DEFAULT_HF_MODEL_ID
from .rlvr_trainer import RlvrTrainerConfig, train_rlvr
from .search import run_search


@dataclass(frozen=True)
class FullPipelineConfig:
    task: str
    model_id: str
    collection_output_dir: str
    adapter_output_dir: str
    population: int
    generations: int
    worlds_per_candidate: int
    mjwarp_episode_steps: int
    mjwarp_policy_iterations: int
    mjwarp_elite_frac: float
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


def run_full_pipeline(config: FullPipelineConfig) -> dict[str, Any]:
    started_at = time.monotonic()
    collection_dir = Path(config.collection_output_dir)
    adapter_dir = Path(config.adapter_output_dir)
    pipeline_state_path = collection_dir / "pipeline_state.json"

    collection_resume = (collection_dir / "checkpoint.json").exists() and not config.overwrite_collection
    results = run_search(
        task=config.task,
        generations=config.generations,
        population=config.population,
        timesteps=1,
        eval_episodes=config.eval_episodes,
        n_envs=1,
        seed=config.seed,
        device=config.device,
        output_dir=collection_dir,
        generator="hf",
        model_id=config.model_id,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        load_in_4bit=config.load_in_4bit,
        sim_backend="mjwarp",
        worlds_per_candidate=config.worlds_per_candidate,
        mjwarp_episode_steps=config.mjwarp_episode_steps,
        mjwarp_policy_iterations=config.mjwarp_policy_iterations,
        mjwarp_elite_frac=config.mjwarp_elite_frac,
        resume=collection_resume,
        overwrite=config.overwrite_collection,
    )

    records_path = collection_dir / "rlvr_records.jsonl"
    if not records_path.exists() or records_path.stat().st_size == 0:
        raise FileNotFoundError(f"Collection did not produce RLVR records at {records_path}")

    metrics_path = adapter_dir / "trainer_metrics.json"
    if metrics_path.exists() and not config.force_train:
        trainer_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        trainer_status = "skipped_existing_adapter"
    else:
        trainer_metrics = train_rlvr(
            RlvrTrainerConfig(
                records_path=str(records_path),
                output_dir=str(adapter_dir),
                model_id=config.model_id,
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
            )
        )
        trainer_status = "trained"

    summary = {
        "config": asdict(config),
        "collection_output_dir": collection_dir.as_posix(),
        "adapter_output_dir": adapter_dir.as_posix(),
        "records_path": records_path.as_posix(),
        "collection_results": len(results),
        "best_verified_reward": max(
            (result.mean_reward for result in results if result.mean_reward is not None),
            default=None,
        ),
        "trainer_status": trainer_status,
        "trainer_final_loss": trainer_metrics.get("final_loss"),
        "elapsed_seconds": time.monotonic() - started_at,
        "updated_at": time.time(),
    }
    collection_dir.mkdir(parents=True, exist_ok=True)
    pipeline_state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full MJWarp EUREKA collection + RLVR training pipeline.")
    parser.add_argument("--task", default="Ant-v5", choices=["Ant-v5"])
    parser.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--collection-output-dir", type=Path, default=Path("runs/deepseek_lite_ant_mjwarp_16x4096"))
    parser.add_argument("--adapter-output-dir", type=Path, default=Path("runs/deepseek_lite_ant_mjwarp_grpo_adapter"))
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--worlds-per-candidate", type=int, default=4096)
    parser.add_argument("--mjwarp-episode-steps", type=int, default=500)
    parser.add_argument("--mjwarp-policy-iterations", type=int, default=4)
    parser.add_argument("--mjwarp-elite-frac", type=float, default=0.1)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--trainer-algorithm", choices=["weighted_sft", "grpo"], default="grpo")
    parser.add_argument("--trainer-epochs", type=int, default=1)
    parser.add_argument("--trainer-batch-size", type=int, default=1)
    parser.add_argument("--trainer-learning-rate", type=float, default=5e-5)
    parser.add_argument("--trainer-max-length", type=int, default=1024)
    parser.add_argument("--trainer-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--trainer-lora-r", type=int, default=16)
    parser.add_argument("--trainer-lora-alpha", type=int, default=32)
    parser.add_argument("--trainer-lora-dropout", type=float, default=0.05)
    parser.add_argument("--trainer-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--trainer-beta-kl", type=float, default=0.01)
    parser.add_argument("--overwrite-collection", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.population < 2 and args.trainer_algorithm == "grpo":
        raise SystemExit("--trainer-algorithm grpo requires --population at least 2")
    config = FullPipelineConfig(
        task=args.task,
        model_id=args.model_id,
        collection_output_dir=args.collection_output_dir.as_posix(),
        adapter_output_dir=args.adapter_output_dir.as_posix(),
        population=args.population,
        generations=args.generations,
        worlds_per_candidate=args.worlds_per_candidate,
        mjwarp_episode_steps=args.mjwarp_episode_steps,
        mjwarp_policy_iterations=args.mjwarp_policy_iterations,
        mjwarp_elite_frac=args.mjwarp_elite_frac,
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
    )
    summary = run_full_pipeline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
