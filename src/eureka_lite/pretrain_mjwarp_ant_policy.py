from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import ANT_TASK
from .cli_options import add_mjwarp_options
from .mjwarp_evaluator import MjwarpEvaluatorConfig, pretrain_original_reward_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain one MJWarp Ant PPO base policy with the original reward.")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/base_ant_mjwarp_policy.pt"))
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    add_mjwarp_options(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MjwarpEvaluatorConfig(
        task=ANT_TASK,
        worlds_per_candidate=args.worlds_per_candidate,
        evaluator=args.mjwarp_evaluator,
        episode_steps=args.mjwarp_episode_steps,
        training_episode_horizon=args.mjwarp_training_episode_horizon,
        policy_iterations=args.mjwarp_policy_iterations,
        ppo_horizon=args.mjwarp_ppo_horizon,
        ppo_epochs=args.mjwarp_ppo_epochs,
        ppo_minibatch_size=args.mjwarp_ppo_minibatch_size,
        ppo_learning_rate=args.mjwarp_ppo_learning_rate,
        elite_frac=args.mjwarp_elite_frac,
        rollout_mode=args.mjwarp_rollout_mode,
        verified_evaluator=args.mjwarp_verified_evaluator,
        verification_steps=args.mjwarp_verification_steps,
        verified_audit_gym=args.mjwarp_verified_audit_gym,
        verified_audit_max_abs_diff=args.mjwarp_verified_audit_max_abs_diff,
        reward_backend=args.mjwarp_reward_backend,
        use_cuda_graph=not args.no_mjwarp_cuda_graph,
        ppo_init_mode="scratch",
        base_policy_checkpoint=None,
        seed=args.seed,
        device=args.device,
        eval_episodes=args.eval_episodes,
    )
    summary = pretrain_original_reward_policy(config, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
