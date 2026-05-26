from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .adapters import ANT_TASK
from .cli_options import add_eureka_options, add_generation_options, add_mjwarp_options, add_negative_sample_options
from .hf_generator import DEFAULT_HF_MODEL_ID
from .search import RunConfig, run_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small EUREKA-inspired reward search.")
    add_eureka_options(parser, population_default=4)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    add_mjwarp_options(parser, include_backend=True)
    add_negative_sample_options(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    parser.add_argument("--generator", default="mock", choices=["mock", "hf"])
    parser.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--adapter-path", default=None)
    add_generation_options(parser)
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/checkpoint.json.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing run artifacts in output-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.population < 1:
        raise SystemExit("--population must be at least 1")
    if args.generations < 1:
        raise SystemExit("--generations must be at least 1")
    if args.eureka_elites < 1:
        raise SystemExit("--eureka-elites must be at least 1")
    if args.negative_rlvr_margin <= 0:
        raise SystemExit("--negative-rlvr-margin must be greater than 0")

    config = RunConfig(
        task=ANT_TASK,
        generations=args.generations,
        population=args.population,
        eureka_elites=args.eureka_elites,
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        generator=args.generator,
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=not args.no_4bit,
        sim_backend=args.sim_backend,
        worlds_per_candidate=args.worlds_per_candidate,
        mjwarp_evaluator=args.mjwarp_evaluator,
        mjwarp_episode_steps=args.mjwarp_episode_steps,
        mjwarp_policy_iterations=args.mjwarp_policy_iterations,
        mjwarp_ppo_horizon=args.mjwarp_ppo_horizon,
        mjwarp_ppo_epochs=args.mjwarp_ppo_epochs,
        mjwarp_ppo_minibatch_size=args.mjwarp_ppo_minibatch_size,
        mjwarp_ppo_learning_rate=args.mjwarp_ppo_learning_rate,
        mjwarp_elite_frac=args.mjwarp_elite_frac,
        include_negative_rlvr_samples=not args.no_negative_rlvr_samples,
        negative_rlvr_margin=args.negative_rlvr_margin,
    )

    console = Console()
    display_config = config
    if args.resume:
        checkpoint_path = args.output_dir / "checkpoint.json"
        if checkpoint_path.exists():
            display_config = RunConfig.from_dict(json.loads(checkpoint_path.read_text(encoding="utf-8"))["config"])
    console.print(
        f"Running reward search: task={display_config.task}, generations={display_config.generations}, "
        f"population={display_config.population}, timesteps={display_config.timesteps}, device={display_config.device}, "
        f"generator={display_config.generator}, sim_backend={display_config.sim_backend}"
    )

    results = run_search(
        config,
        output_dir=args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    table = Table(title="Reward Candidates")
    table.add_column("Rank", justify="right")
    table.add_column("Mean Reward", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Candidate")

    for rank, result in enumerate(
        sorted(results, key=lambda item: item.mean_reward if item.mean_reward is not None else float("-inf"), reverse=True),
        start=1,
    ):
        table.add_row(
            str(rank),
            "n/a" if result.mean_reward is None else f"{result.mean_reward:.1f}",
            "n/a" if result.std_reward is None else f"{result.std_reward:.1f}",
            result.candidate.name,
        )

    console.print(table)
    console.print(f"Wrote results to {args.output_dir}")
