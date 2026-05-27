from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import ANT_TASK
from .cli_options import add_mjwarp_options
from .generators import initial_population
from .mjwarp_evaluator import MjwarpEvaluatorConfig, train_and_evaluate_mjwarp_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Ant PPO budget without running RLVR model updates.")
    parser.add_argument("--output", type=Path, default=Path("runs/calibration/mjwarp_ppo_budget.json"))
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 24, 48, 96])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    add_mjwarp_options(parser)
    return parser.parse_args()


def rank_correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 1.0
    left_ranks = np.argsort(np.argsort(np.asarray(left, dtype=np.float64)))
    right_ranks = np.argsort(np.argsort(np.asarray(right, dtype=np.float64)))
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def main() -> None:
    args = parse_args()
    candidates = initial_population(ANT_TASK, args.population, random.Random(args.seeds[0]))
    reports: list[dict[str, Any]] = []
    by_seed: dict[int, dict[int, list[float]]] = {}
    for seed in args.seeds:
        by_seed[seed] = {}
        for budget in args.budgets:
            config = MjwarpEvaluatorConfig(
                task=ANT_TASK,
                worlds_per_candidate=args.worlds_per_candidate,
                episode_steps=args.mjwarp_episode_steps,
                training_episode_horizon=args.mjwarp_training_episode_horizon,
                policy_iterations=budget,
                ppo_horizon=args.mjwarp_ppo_horizon,
                ppo_epochs=args.mjwarp_ppo_epochs,
                ppo_minibatch_size=args.mjwarp_ppo_minibatch_size,
                ppo_learning_rate=args.mjwarp_ppo_learning_rate,
                ppo_init_mode=args.mjwarp_ppo_init_mode,
                base_policy_checkpoint=args.mjwarp_base_policy_checkpoint,
                rollout_mode=args.mjwarp_rollout_mode,
                verified_evaluator=args.mjwarp_verified_evaluator,
                verification_steps=args.mjwarp_verification_steps,
                verified_audit_gym=args.mjwarp_verified_audit_gym,
                verified_audit_max_abs_diff=args.mjwarp_verified_audit_max_abs_diff,
                reward_backend=args.mjwarp_reward_backend,
                use_cuda_graph=not args.no_mjwarp_cuda_graph,
                seed=seed,
                device=args.device,
                eval_episodes=args.eval_episodes,
            )
            started_at = time.monotonic()
            results = train_and_evaluate_mjwarp_batch(candidates, config)
            scores = [float(row["mean_reward"]) for row in results]
            by_seed[seed][budget] = scores
            reports.append(
                {
                    "seed": seed,
                    "policy_iterations": budget,
                    "elapsed_seconds": time.monotonic() - started_at,
                    "scores": scores,
                    "best_candidate_index": int(np.argmax(scores)),
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "config": asdict(config),
                }
            )
    largest_budget = max(args.budgets)
    comparisons = [
        {
            "seed": seed,
            "policy_iterations": budget,
            "reference_policy_iterations": largest_budget,
            "rank_correlation": rank_correlation(scores, by_seed[seed][largest_budget]),
            "same_best_candidate": int(np.argmax(scores)) == int(np.argmax(by_seed[seed][largest_budget])),
        }
        for seed, budgets in by_seed.items()
        for budget, scores in budgets.items()
        if budget != largest_budget
    ]
    payload = {
        "task": ANT_TASK,
        "candidate_names": [candidate.name for candidate in candidates],
        "runs": reports,
        "comparisons_to_largest_budget": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output.as_posix(), "runs": len(reports), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
