from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .adapters import ADAPTERS
from .hf_generator import DEFAULT_HF_MODEL_ID
from .search import run_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small EUREKA-inspired reward search.")
    parser.add_argument(
        "--task",
        default="Ant-v5",
        choices=sorted(ADAPTERS),
        help="Gymnasium task. Ant-v5 is the local EUREKA-style default.",
    )
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    parser.add_argument("--generator", default="mock", choices=["mock", "hf"])
    parser.add_argument("--model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized HF model loading.")
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/checkpoint.json.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing run artifacts in output-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.population < 1:
        raise SystemExit("--population must be at least 1")
    if args.generations < 1:
        raise SystemExit("--generations must be at least 1")

    console = Console()
    display_args = args
    if args.resume:
        checkpoint_path = args.output_dir / "checkpoint.json"
        if checkpoint_path.exists():
            config = json.loads(checkpoint_path.read_text(encoding="utf-8"))["config"]
            for key, value in config.items():
                setattr(display_args, key, value)
    console.print(
        f"Running reward search: task={display_args.task}, generations={display_args.generations}, "
        f"population={display_args.population}, timesteps={display_args.timesteps}, device={display_args.device}, "
        f"generator={display_args.generator}"
    )

    results = run_search(
        task=args.task,
        generations=args.generations,
        population=args.population,
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        generator=args.generator,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=not args.no_4bit,
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
