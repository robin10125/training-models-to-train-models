from __future__ import annotations

import argparse


def add_eureka_options(parser: argparse.ArgumentParser, *, population_default: int) -> None:
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=population_default)
    parser.add_argument("--eureka-elites", type=int, default=4)


def add_mjwarp_options(parser: argparse.ArgumentParser, *, include_backend: bool = False) -> None:
    if include_backend:
        parser.add_argument("--sim-backend", default="sb3", choices=["sb3", "mjwarp"])
    parser.add_argument("--worlds-per-candidate", type=int, default=4096)
    parser.add_argument("--mjwarp-evaluator", default="ppo", choices=["ppo", "search"])
    parser.add_argument("--mjwarp-episode-steps", type=int, default=500)
    parser.add_argument(
        "--mjwarp-training-episode-horizon",
        type=int,
        default=1000,
        help="Maximum Ant episode length during PPO training; completed worlds reset in place.",
    )
    parser.add_argument("--mjwarp-policy-iterations", type=int, default=96)
    parser.add_argument("--mjwarp-ppo-horizon", type=int, default=32)
    parser.add_argument("--mjwarp-ppo-epochs", type=int, default=4)
    parser.add_argument("--mjwarp-ppo-minibatch-size", type=int, default=16_384)
    parser.add_argument("--mjwarp-ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--mjwarp-elite-frac", type=float, default=0.1)
    parser.add_argument("--mjwarp-rollout-mode", choices=["gpu", "host"], default="gpu")
    parser.add_argument("--mjwarp-verified-evaluator", choices=["mjwarp", "gym"], default="mjwarp")
    parser.add_argument("--mjwarp-verification-steps", type=int, default=1000)
    parser.add_argument(
        "--mjwarp-verified-audit-gym",
        action="store_true",
        help="Also score MJWarp-verified policies with Gym Ant-v5 and record agreement.",
    )
    parser.add_argument(
        "--mjwarp-verified-audit-max-abs-diff",
        type=float,
        default=None,
        help="Fail if a Gym audit episode differs from MJWarp by more than this threshold.",
    )
    parser.add_argument(
        "--mjwarp-reward-backend",
        choices=["eager", "compiled"],
        default="eager",
        help="Reward-expression execution backend; compiled falls back to eager on compile failure.",
    )
    parser.add_argument(
        "--no-mjwarp-candidate-batching",
        action="store_true",
        help="Evaluate MJWarp PPO reward candidates sequentially instead of batching them on one GPU.",
    )
    parser.add_argument(
        "--no-mjwarp-cuda-graph",
        action="store_true",
        help="Disable CUDA-graph replay for the MuJoCo Warp physics substeps.",
    )


def add_negative_sample_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-negative-rlvr-samples",
        action="store_true",
        help="Exclude invalid code attempts and failed evaluations from RLVR training.",
    )
    parser.add_argument(
        "--negative-rlvr-margin",
        type=float,
        default=1.0,
        help="Penalty margin below the worst successful candidate for negative RLVR samples.",
    )


def add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized HF model loading.")
