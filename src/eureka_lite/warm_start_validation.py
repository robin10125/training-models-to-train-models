from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .adapters import ANT_TASK
from .generators import initial_population
from .mjwarp_evaluator import (
    AntActorCritic,
    MjwarpEvaluatorConfig,
    evaluate_policy_in_mjwarp,
    load_base_policy_checkpoint,
    load_base_policy_into_single,
    pretrain_original_reward_policy,
    seed_torch_policy_rng,
    train_and_evaluate_mjwarp_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate whether MJWarp Ant PPO warm starts are useful.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rtx2070_warm_start_validation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--worlds-per-candidate", type=int, default=512)
    parser.add_argument("--episode-steps", type=int, default=128)
    parser.add_argument("--training-episode-horizon", type=int, default=1000)
    parser.add_argument("--base-policy-iterations", type=int, default=64)
    parser.add_argument("--candidate-policy-iterations", type=int, default=8)
    parser.add_argument("--ppo-horizon", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-minibatch-size", type=int, default=4096)
    parser.add_argument("--ppo-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--init-std", type=float, default=0.35)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--verification-steps", type=int, default=1000)
    parser.add_argument("--acceptance-margin", type=float, default=25.0)
    parser.add_argument("--reuse-base-policy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "base_policy.pt"
    started_at = time.monotonic()

    common = {
        "worlds_per_candidate": args.worlds_per_candidate,
        "episode_steps": args.episode_steps,
        "training_episode_horizon": args.training_episode_horizon,
        "ppo_horizon": args.ppo_horizon,
        "ppo_epochs": args.ppo_epochs,
        "ppo_minibatch_size": args.ppo_minibatch_size,
        "ppo_learning_rate": args.ppo_learning_rate,
        "init_std": args.init_std,
        "verified_evaluator": "mjwarp",
        "verification_steps": args.verification_steps,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "device": args.device,
    }

    if args.reuse_base_policy and checkpoint.exists():
        print("reusing_base_policy", flush=True)
        checkpoint_payload = load_base_policy_checkpoint(checkpoint, map_location="cpu")
        checkpoint_metadata = checkpoint_payload.get("metadata", {})
        pretrain = {
            "output": checkpoint.as_posix(),
            "mean_verified_return": checkpoint_metadata.get("mean_verified_return"),
            "std_verified_return": checkpoint_metadata.get("std_verified_return"),
            "episode_rewards": checkpoint_metadata.get("episode_rewards"),
            "elapsed_seconds": None,
            "reused": True,
        }
    else:
        print("pretraining_base_policy", flush=True)
        pretrain_config = MjwarpEvaluatorConfig(
            policy_iterations=args.base_policy_iterations,
            ppo_init_mode="scratch",
            base_policy_checkpoint=None,
            **common,
        )
        pretrain = pretrain_original_reward_policy(pretrain_config, checkpoint)

    print("evaluating_policy_baselines", flush=True)
    policy_baselines = evaluate_policy_baselines(checkpoint, args)

    print("comparing_candidate_training", flush=True)
    candidates = initial_population(ANT_TASK, args.population, random.Random(args.seed))
    comparisons = []
    for mode in ("scratch", "base"):
        config = MjwarpEvaluatorConfig(
            policy_iterations=args.candidate_policy_iterations,
            ppo_init_mode=mode,
            base_policy_checkpoint=checkpoint.as_posix() if mode == "base" else None,
            **common,
        )
        mode_started = time.monotonic()
        rows = train_and_evaluate_mjwarp_batch(candidates, config)
        comparisons.append(
            {
                "mode": mode,
                "elapsed_seconds": time.monotonic() - mode_started,
                "mean_of_candidate_means": float(np.mean([row["mean_reward"] for row in rows])),
                "best_candidate_mean": float(max(row["mean_reward"] for row in rows)),
                "rows": [
                    {
                        "candidate": candidates[index].name,
                        "mean_reward": row["mean_reward"],
                        "std_reward": row["std_reward"],
                        "episode_rewards": row["episode_rewards"],
                        "ppo_init_mode": row["metadata"].get("ppo_init_mode"),
                        "base_policy_verified_return": row["metadata"].get("base_policy_verified_return"),
                        "candidate_finetune_budget": row["metadata"].get("candidate_finetune_budget"),
                    }
                    for index, row in enumerate(rows)
                ],
            }
        )

    random_mean = policy_baselines["random_policy"]["mean_reward"]
    zero_mean = policy_baselines["zero_policy"]["mean_reward"]
    base_mean = policy_baselines["base_policy"]["mean_reward"]
    acceptance = {
        "base_beats_random_by_margin": base_mean > random_mean + args.acceptance_margin,
        "base_beats_zero_by_margin": base_mean > zero_mean + args.acceptance_margin,
        "acceptance_margin": args.acceptance_margin,
        "base_minus_random": base_mean - random_mean,
        "base_minus_zero": base_mean - zero_mean,
    }

    payload = {
        "config": serializable_args(args),
        "checkpoint": checkpoint.as_posix(),
        "pretrain": pretrain,
        "policy_baselines": policy_baselines,
        "candidate_comparisons": comparisons,
        "acceptance": acceptance,
        "elapsed_seconds": time.monotonic() - started_at,
    }
    output = args.output_dir / "warm_start_validation.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": output.as_posix(), "acceptance": acceptance}, indent=2), flush=True)


def evaluate_policy_baselines(checkpoint: Path, args: argparse.Namespace) -> dict[str, Any]:
    import mujoco_warp as mjw
    import torch
    import warp as wp

    env = gym.make(ANT_TASK)
    try:
        mjm = env.unwrapped.model
        frame_skip = int(env.unwrapped.frame_skip)
        dt = float(mjm.opt.timestep * frame_skip)
        action_dim = int(mjm.nu)
        obs_dim = int(mjm.nq - 2 + mjm.nv)
        torch_device = torch.device(args.device)
        wp.init()
        with wp.ScopedDevice(args.device):
            model = mjw.put_model(mjm)

            class ZeroPolicy:
                def mean_action(self, obs: Any) -> Any:
                    return torch.zeros((obs.shape[0], action_dim), dtype=torch.float32, device=obs.device)

            seed_torch_policy_rng(args.seed + 1)
            random_policy = AntActorCritic(obs_dim, action_dim).to(torch_device)
            random_policy.eval()
            base_policy = AntActorCritic(obs_dim, action_dim).to(torch_device)
            load_base_policy_into_single(
                base_policy,
                checkpoint,
                obs_dim=obs_dim,
                action_dim=action_dim,
                torch_device=torch_device,
            )
            base_policy.eval()
            policies = {
                "zero_policy": ZeroPolicy(),
                "random_policy": random_policy,
                "base_policy": base_policy,
            }
            results = {}
            for name, policy in policies.items():
                data = mjw.make_data(mjm, nworld=args.eval_episodes)
                wp.synchronize()
                returns = evaluate_policy_in_mjwarp(
                    task=ANT_TASK,
                    model=model,
                    data=data,
                    mjm=mjm,
                    policy=("ppo", policy),
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    eval_episodes=args.eval_episodes,
                    episode_steps=args.verification_steps,
                    seed=args.seed + 10_000,
                    device=args.device,
                    dt=dt,
                    frame_skip=frame_skip,
                    use_cuda_graph=True,
                    wp=wp,
                    mjw=mjw,
                )
                results[name] = {
                    "mean_reward": float(np.mean(returns)),
                    "std_reward": float(np.std(returns)),
                    "episode_rewards": returns,
                }
            return results
    finally:
        env.close()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    row = vars(args).copy()
    for key, value in row.items():
        if isinstance(value, Path):
            row[key] = value.as_posix()
    return row


if __name__ == "__main__":
    main()
