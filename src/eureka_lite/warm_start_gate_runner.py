from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .mjwarp_evaluator import MjwarpEvaluatorConfig, pretrain_original_reward_policy
from .warm_start_validation import evaluate_policy_baselines, serializable_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MJWarp Ant warm-start policy in stages until a validation gate is met."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rtx2070_warm_start_gate"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2070)
    parser.add_argument("--worlds-per-candidate", type=int, default=256)
    parser.add_argument("--episode-steps", type=int, default=128)
    parser.add_argument("--training-episode-horizon", type=int, default=1000)
    parser.add_argument("--stage-policy-iterations", type=int, default=256)
    parser.add_argument("--ppo-horizon", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-minibatch-size", type=int, default=4096)
    parser.add_argument("--ppo-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--init-std", type=float, default=0.35)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--verification-steps", type=int, default=1000)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--plateau-hours", type=float, default=None)
    parser.add_argument("--plateau-tolerance", type=float, default=25.0)
    parser.add_argument("--gate-min-return", type=float, default=2500.0)
    parser.add_argument("--gate-random-margin", type=float, default=1000.0)
    parser.add_argument("--gate-zero-margin", type=float, default=1000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = args.output_dir / "base_policy.pt"
    best_checkpoint = args.output_dir / "best_base_policy.pt"
    status_path = args.output_dir / "warm_start_gate_status.json"
    history_path = args.output_dir / "warm_start_gate_history.json"
    started_at = time.monotonic()
    history: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None
    stage_index = 0

    while True:
        now = time.monotonic()
        elapsed = now - started_at
        if elapsed >= args.max_hours * 3600.0:
            reason = "max_hours_reached"
            break

        stage_index += 1
        init_mode = "base" if latest_checkpoint.exists() else "scratch"
        checkpoint_path = latest_checkpoint.as_posix() if latest_checkpoint.exists() else None
        print(
            json.dumps(
                {
                    "event": "stage_start",
                    "stage": stage_index,
                    "elapsed_seconds": elapsed,
                    "ppo_init_mode": init_mode,
                    "checkpoint": checkpoint_path,
                }
            ),
            flush=True,
        )
        pretrain_config = MjwarpEvaluatorConfig(
            worlds_per_candidate=args.worlds_per_candidate,
            episode_steps=args.episode_steps,
            training_episode_horizon=args.training_episode_horizon,
            policy_iterations=args.stage_policy_iterations,
            ppo_horizon=args.ppo_horizon,
            ppo_epochs=args.ppo_epochs,
            ppo_minibatch_size=args.ppo_minibatch_size,
            ppo_learning_rate=args.ppo_learning_rate,
            init_std=args.init_std,
            verified_evaluator="mjwarp",
            verification_steps=args.verification_steps,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            device=args.device,
            ppo_init_mode=init_mode,
            base_policy_checkpoint=checkpoint_path,
        )
        stage_started = time.monotonic()
        pretrain = pretrain_original_reward_policy(pretrain_config, latest_checkpoint)
        baselines = evaluate_policy_baselines(latest_checkpoint, args)
        entry = build_history_entry(
            stage=stage_index,
            stage_started=stage_started,
            pretrain=pretrain,
            baselines=baselines,
            args=args,
        )
        history.append(entry)

        if best_entry is None or entry["base_policy_mean_reward"] > best_entry["base_policy_mean_reward"]:
            best_entry = entry
            shutil.copy2(latest_checkpoint, best_checkpoint)
            entry["saved_as_best"] = True
        else:
            entry["saved_as_best"] = False

        status = {
            "config": serializable_args(args),
            "latest_checkpoint": latest_checkpoint.as_posix(),
            "best_checkpoint": best_checkpoint.as_posix() if best_checkpoint.exists() else None,
            "best_stage": None if best_entry is None else best_entry["stage"],
            "best_base_policy_mean_reward": None if best_entry is None else best_entry["base_policy_mean_reward"],
            "latest_stage": stage_index,
            "elapsed_seconds": time.monotonic() - started_at,
            "latest_entry": entry,
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        history_path.write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
        print(json.dumps({"event": "stage_complete", "entry": entry}, indent=2), flush=True)

        if gate_passed(entry):
            reason = "gate_surpassed"
            break
        if args.plateau_hours is not None and plateau_or_decrease(
            history,
            plateau_window_seconds=args.plateau_hours * 3600.0,
            tolerance=args.plateau_tolerance,
        ):
            reason = "plateau_or_decrease_for_window"
            break

    final_payload = {
        "config": serializable_args(args),
        "stop_reason": reason,
        "elapsed_seconds": time.monotonic() - started_at,
        "latest_checkpoint": latest_checkpoint.as_posix() if latest_checkpoint.exists() else None,
        "best_checkpoint": best_checkpoint.as_posix() if best_checkpoint.exists() else None,
        "best_stage": None if best_entry is None else best_entry["stage"],
        "best_base_policy_mean_reward": None if best_entry is None else best_entry["base_policy_mean_reward"],
        "history_count": len(history),
    }
    status_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    print(json.dumps({"event": "run_complete", **final_payload}, indent=2), flush=True)


def build_history_entry(
    *,
    stage: int,
    stage_started: float,
    pretrain: dict[str, Any],
    baselines: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    zero_mean = float(baselines["zero_policy"]["mean_reward"])
    random_mean = float(baselines["random_policy"]["mean_reward"])
    base_mean = float(baselines["base_policy"]["mean_reward"])
    return {
        "stage": stage,
        "wall_time": time.time(),
        "stage_elapsed_seconds": time.monotonic() - stage_started,
        "pretrain_mean_verified_return": pretrain.get("mean_verified_return"),
        "pretrain_std_verified_return": pretrain.get("std_verified_return"),
        "base_policy_mean_reward": base_mean,
        "base_policy_std_reward": float(baselines["base_policy"]["std_reward"]),
        "random_policy_mean_reward": random_mean,
        "zero_policy_mean_reward": zero_mean,
        "base_minus_random": base_mean - random_mean,
        "base_minus_zero": base_mean - zero_mean,
        "gate_min_return": args.gate_min_return,
        "gate_random_margin": args.gate_random_margin,
        "gate_zero_margin": args.gate_zero_margin,
        "gate_passed": (
            base_mean >= args.gate_min_return
            and (base_mean - random_mean) >= args.gate_random_margin
            and (base_mean - zero_mean) >= args.gate_zero_margin
        ),
        "policy_baselines": baselines,
    }


def gate_passed(entry: dict[str, Any]) -> bool:
    return bool(entry["gate_passed"])


def plateau_or_decrease(
    history: list[dict[str, Any]],
    *,
    plateau_window_seconds: float,
    tolerance: float,
) -> bool:
    if len(history) < 2:
        return False
    latest = history[-1]
    cutoff = float(latest["wall_time"]) - plateau_window_seconds
    window = [entry for entry in history if float(entry["wall_time"]) >= cutoff]
    if len(window) < 2:
        return False
    scores = [float(entry["base_policy_mean_reward"]) for entry in window]
    if max(scores) - min(scores) <= tolerance:
        return True
    nonincreasing = True
    for previous, current in zip(scores, scores[1:], strict=True):
        if current > previous + tolerance:
            nonincreasing = False
            break
    return nonincreasing


if __name__ == "__main__":
    main()
