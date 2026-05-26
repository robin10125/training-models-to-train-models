from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .adapters import ANT_TASK


@dataclass(frozen=True)
class MjwarpAntConfig:
    worlds: int = 1024
    steps: int = 1000
    warmup_steps: int = 10
    device: str = "cuda:0"
    action_mode: str = "random-once"
    seed: int = 7
    nconmax: int | None = None
    njmax: int | None = None
    use_cuda_graph: bool = True


def run_mjwarp_ant(config: MjwarpAntConfig) -> dict[str, Any]:
    """Run many Ant worlds in parallel with MuJoCo Warp.

    This is intentionally a simulation-throughput runner, not an SB3 VecEnv.
    It gives the project a concrete GPU batched-physics path that can later be
    used by a custom policy rollout/training loop.
    """

    if config.worlds < 1:
        raise ValueError("worlds must be at least 1")
    if config.steps < 1:
        raise ValueError("steps must be at least 1")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.action_mode not in {"zero", "random-once", "random-each-step"}:
        raise ValueError("action_mode must be one of: zero, random-once, random-each-step")
    if config.use_cuda_graph and config.action_mode == "random-each-step":
        raise ValueError("CUDA graph capture requires a fixed action buffer; use zero or random-once")

    try:
        import gymnasium as gym
        import mujoco_warp as mjw
        import warp as wp
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo Warp support requires optional GPU dependencies. Install with "
            "`pip install '.[mjwarp]'` or `pip install mujoco-warp`."
        ) from exc

    wp.init()
    rng = np.random.default_rng(config.seed)

    env = gym.make(ANT_TASK)
    try:
        mjm = env.unwrapped.model
        nu = int(mjm.nu)

        with wp.ScopedDevice(config.device):
            model = mjw.put_model(mjm)
            data = mjw.make_data(
                mjm,
                nworld=config.worlds,
                nconmax=config.nconmax,
                njmax=config.njmax,
            )
            mjw.reset_data(model, data)

            if config.action_mode == "zero":
                ctrl = np.zeros((config.worlds, nu), dtype=np.float32)
            else:
                ctrl = rng.uniform(-1.0, 1.0, size=(config.worlds, nu)).astype(np.float32)
            wp.copy(data.ctrl, wp.array(ctrl, dtype=wp.float32, device=config.device))

            for _ in range(config.warmup_steps):
                if config.action_mode == "random-each-step":
                    ctrl = rng.uniform(-1.0, 1.0, size=(config.worlds, nu)).astype(np.float32)
                    wp.copy(data.ctrl, wp.array(ctrl, dtype=wp.float32, device=config.device))
                mjw.step(model, data)
            wp.synchronize()

            graph = None
            if config.use_cuda_graph:
                with wp.ScopedCapture(device=config.device) as capture:
                    mjw.step(model, data)
                graph = capture.graph

            started_at = time.perf_counter()
            for _ in range(config.steps):
                if config.action_mode == "random-each-step":
                    ctrl = rng.uniform(-1.0, 1.0, size=(config.worlds, nu)).astype(np.float32)
                    wp.copy(data.ctrl, wp.array(ctrl, dtype=wp.float32, device=config.device))
                if graph is None:
                    mjw.step(model, data)
                else:
                    wp.capture_launch(graph)
            wp.synchronize()
            elapsed_seconds = time.perf_counter() - started_at

            sample_qpos = np.asarray(data.qpos.numpy()[0], dtype=float).tolist()
    finally:
        env.close()

    world_steps = config.worlds * config.steps
    return {
        "config": asdict(config),
        "world_steps": world_steps,
        "elapsed_seconds": elapsed_seconds,
        "world_steps_per_second": world_steps / elapsed_seconds if elapsed_seconds > 0 else float("inf"),
        "sample_qpos_world0": sample_qpos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched Ant-v5 physics with MuJoCo Warp.")
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-mode", choices=["zero", "random-once", "random-each-step"], default="random-once")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--no-cuda-graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_mjwarp_ant(
        MjwarpAntConfig(
            worlds=args.worlds,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            device=args.device,
            action_mode=args.action_mode,
            seed=args.seed,
            nconmax=args.nconmax,
            njmax=args.njmax,
            use_cuda_graph=not args.no_cuda_graph,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
