from __future__ import annotations

import ast
import builtins
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .adapters import get_adapter
from .rewards import (
    ALLOWED_FUNCS,
    RewardCandidate,
    RewardExpression,
    total_expression_from_components,
    validate_component_expressions,
)


@dataclass(frozen=True)
class MjwarpEvaluatorConfig:
    task: str = "Ant-v5"
    evaluator: str = "ppo"
    worlds_per_candidate: int = 4096
    episode_steps: int = 500
    training_episode_horizon: int = 1000
    policy_iterations: int = 96
    ppo_horizon: int = 32
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 16_384
    ppo_learning_rate: float = 3.0e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_value_coef: float = 2.0
    ppo_entropy_coef: float = 0.0
    ppo_max_grad_norm: float = 1.0
    rollout_mode: str = "gpu"
    verified_evaluator: str = "mjwarp"
    verification_steps: int = 1000
    verified_audit_gym: bool = False
    verified_audit_max_abs_diff: float | None = None
    reward_backend: str = "eager"
    use_cuda_graph: bool = True
    elite_frac: float = 0.1
    init_std: float = 0.35
    min_std: float = 0.03
    ppo_init_mode: str = "base"
    base_policy_checkpoint: str | None = "checkpoints/base_ant_mjwarp_policy.pt"
    seed: int = 7
    device: str = "cuda:0"
    eval_episodes: int = 5


def validate_verification_audit_config(config: MjwarpEvaluatorConfig) -> None:
    if config.verified_audit_gym and config.verified_evaluator != "mjwarp":
        raise ValueError("verified_audit_gym requires verified_evaluator='mjwarp'")
    if config.verified_audit_gym and config.verification_steps != 1000:
        raise ValueError("verified_audit_gym requires verification_steps=1000 to match Ant-v5")
    if config.verified_audit_max_abs_diff is not None and not config.verified_audit_gym:
        raise ValueError("verified_audit_max_abs_diff requires verified_audit_gym")
    if config.verified_audit_max_abs_diff is not None and config.verified_audit_max_abs_diff < 0:
        raise ValueError("verified_audit_max_abs_diff must be non-negative")


def validate_reward_backend(backend: str) -> None:
    if backend not in {"eager", "compiled"}:
        raise ValueError("reward_backend must be one of: eager, compiled")


def validate_ppo_init_config(config: MjwarpEvaluatorConfig) -> None:
    if config.ppo_init_mode not in {"scratch", "base"}:
        raise ValueError("ppo_init_mode must be one of: scratch, base")
    if config.ppo_init_mode == "base":
        if not config.base_policy_checkpoint:
            raise ValueError("base_policy_checkpoint is required when ppo_init_mode='base'")
        if config.evaluator != "ppo":
            raise ValueError("base PPO initialization requires evaluator='ppo'")


def base_policy_metadata(config: MjwarpEvaluatorConfig) -> dict[str, Any]:
    metadata = {
        "ppo_init_mode": config.ppo_init_mode,
        "base_policy_checkpoint": config.base_policy_checkpoint,
        "candidate_finetune_budget": config.worlds_per_candidate * config.episode_steps * config.policy_iterations,
    }
    if config.ppo_init_mode == "base" and config.base_policy_checkpoint:
        checkpoint_metadata = read_base_policy_checkpoint_metadata(config.base_policy_checkpoint)
        metadata.update(
            {
                "base_policy_verified_return": checkpoint_metadata.get("mean_verified_return"),
                "base_policy_training_budget": checkpoint_metadata.get("training_world_steps"),
            }
        )
    return metadata


def read_base_policy_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    import torch

    checkpoint = load_base_policy_checkpoint(path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def build_verification_audit(
    primary_returns: list[float],
    gym_reference_returns: list[float] | None,
    config: MjwarpEvaluatorConfig,
) -> dict[str, Any] | None:
    if gym_reference_returns is None:
        return None
    differences = [
        abs(float(primary) - float(reference))
        for primary, reference in zip(primary_returns, gym_reference_returns, strict=True)
    ]
    max_abs_diff = max(differences, default=0.0)
    audit = {
        "reference_evaluator": "gym",
        "gym_episode_rewards": gym_reference_returns,
        "absolute_differences": differences,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": float(np.mean(differences)) if differences else 0.0,
        "passed": config.verified_audit_max_abs_diff is None
        or max_abs_diff <= config.verified_audit_max_abs_diff,
        "threshold": config.verified_audit_max_abs_diff,
    }
    if not audit["passed"]:
        raise RuntimeError(
            "MJWarp verified-return audit failed: "
            f"max_abs_diff={max_abs_diff:.6g} exceeds threshold={config.verified_audit_max_abs_diff:.6g}"
        )
    return audit


def train_and_evaluate_mjwarp(candidate: RewardCandidate, config: MjwarpEvaluatorConfig) -> dict[str, Any]:
    if config.task != "Ant-v5":
        raise ValueError("The MJWarp evaluator currently supports only Ant-v5")
    if config.worlds_per_candidate < 2:
        raise ValueError("worlds_per_candidate must be at least 2")
    if config.episode_steps < 1:
        raise ValueError("episode_steps must be at least 1")
    if config.policy_iterations < 1:
        raise ValueError("policy_iterations must be at least 1")
    if config.training_episode_horizon < 1:
        raise ValueError("training_episode_horizon must be at least 1")
    if config.evaluator not in {"ppo", "search"}:
        raise ValueError("evaluator must be one of: ppo, search")
    if config.rollout_mode not in {"gpu", "host"}:
        raise ValueError("rollout_mode must be one of: gpu, host")
    if config.verified_evaluator not in {"mjwarp", "gym"}:
        raise ValueError("verified_evaluator must be one of: mjwarp, gym")
    if config.verification_steps < 1:
        raise ValueError("verification_steps must be at least 1")
    validate_verification_audit_config(config)
    validate_reward_backend(config.reward_backend)
    validate_ppo_init_config(config)
    if config.ppo_horizon < 1:
        raise ValueError("ppo_horizon must be at least 1")
    if not 0.0 < config.elite_frac <= 1.0:
        raise ValueError("elite_frac must be in (0, 1]")

    try:
        import mujoco_warp as mjw
        import warp as wp
    except ImportError as exc:
        raise RuntimeError(
            "The MJWarp evaluator requires optional GPU dependencies. Install with "
            "`pip install -e '.[mjwarp]'`."
        ) from exc

    started_at = time.monotonic()
    adapter = get_adapter(config.task)
    reward_program = VectorizedRewardProgram(
        component_expressions=candidate.component_expressions,
        expression=candidate.expression,
        allowed_names=adapter.reward_variables,
    )
    rng = np.random.default_rng(config.seed)

    env = gym.make(config.task)
    try:
        mjm = env.unwrapped.model
        frame_skip = int(env.unwrapped.frame_skip)
        dt = float(mjm.opt.timestep * frame_skip)
        eval_episode_steps = config.verification_steps
        action_dim = int(mjm.nu)
        obs_dim = int(mjm.nq - 2 + mjm.nv)
        wp.init()
        with wp.ScopedDevice(config.device):
            model = mjw.put_model(mjm)
            data = mjw.make_data(mjm, nworld=config.worlds_per_candidate)
            wp.synchronize()
            if config.evaluator == "ppo":
                policy, best_shaped_return, iteration_summaries = train_ppo_policy(
                    model=model,
                    data=data,
                    mjm=mjm,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    reward_program=reward_program,
                    config=config,
                    dt=dt,
                    frame_skip=frame_skip,
                    wp=wp,
                    mjw=mjw,
                )
                eval_policy = ("ppo", policy)
            else:
                best_params, best_shaped_return, iteration_summaries = train_search_policy(
                    model=model,
                    data=data,
                    mjm=mjm,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    reward_program=reward_program,
                    config=config,
                    dt=dt,
                    frame_skip=frame_skip,
                    rng=rng,
                    wp=wp,
                    mjw=mjw,
                )
                eval_policy = ("linear", best_params)
    finally:
        env.close()

    audit_returns: list[float] | None = None
    if config.verified_evaluator == "mjwarp":
        with wp.ScopedDevice(config.device):
            eval_data = mjw.make_data(mjm, nworld=config.eval_episodes)
            wp.synchronize()
            eval_returns = evaluate_policy_in_mjwarp(
                task=config.task,
                model=model,
                data=eval_data,
                mjm=mjm,
                policy=eval_policy,
                obs_dim=obs_dim,
                action_dim=action_dim,
                eval_episodes=config.eval_episodes,
                episode_steps=eval_episode_steps,
                seed=config.seed + 10_000,
                device=config.device,
                dt=dt,
                frame_skip=frame_skip,
                use_cuda_graph=config.use_cuda_graph,
                wp=wp,
                mjw=mjw,
            )
        if config.verified_audit_gym:
            audit_returns = evaluate_policy_in_gym(
                task=config.task,
                policy=eval_policy,
                obs_dim=obs_dim,
                action_dim=action_dim,
                eval_episodes=config.eval_episodes,
                seed=config.seed + 10_000,
            )
    else:
        eval_returns = evaluate_policy_in_gym(
            task=config.task,
            policy=eval_policy,
            obs_dim=obs_dim,
            action_dim=action_dim,
            eval_episodes=config.eval_episodes,
            seed=config.seed + 10_000,
        )
    verification_audit = build_verification_audit(eval_returns, audit_returns, config)
    return {
        "mean_reward": float(np.mean(eval_returns)),
        "std_reward": float(np.std(eval_returns)),
        "episode_rewards": eval_returns,
        "elapsed_seconds": time.monotonic() - started_at,
        "metadata": {
            "sim_backend": "mjwarp",
            "mjwarp_evaluator": config.evaluator,
            "worlds_per_candidate": config.worlds_per_candidate,
            "episode_steps": config.episode_steps,
            "training_episode_horizon": config.training_episode_horizon,
            "policy_iterations": config.policy_iterations,
            "training_world_steps": config.worlds_per_candidate * config.episode_steps * config.policy_iterations,
            "ppo_horizon": config.ppo_horizon,
            "ppo_epochs": config.ppo_epochs,
            "ppo_minibatch_size": config.ppo_minibatch_size,
            "ppo_hidden_sizes": [256, 128, 64],
            **base_policy_metadata(config),
            "rollout_mode": config.rollout_mode,
            "verified_evaluator": config.verified_evaluator,
            "verified_audit": verification_audit,
            "reward_backend": config.reward_backend,
            "candidate_batching": False,
            "cuda_graph_requested": config.use_cuda_graph,
            "frame_skip": frame_skip,
            "evaluation_episode_steps": eval_episode_steps,
            "elite_frac": config.elite_frac,
            "reward_components": reward_program.component_expressions,
            "best_shaped_return": best_shaped_return,
            "iteration_summaries": iteration_summaries,
        },
    }


def train_and_evaluate_mjwarp_batch(
    candidates: list[RewardCandidate], config: MjwarpEvaluatorConfig
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if config.task != "Ant-v5":
        raise ValueError("The MJWarp evaluator currently supports only Ant-v5")
    if config.evaluator != "ppo" or config.rollout_mode != "gpu":
        raise ValueError("Candidate batching requires the GPU PPO evaluator")
    if config.worlds_per_candidate < 2 or config.episode_steps < 1 or config.policy_iterations < 1:
        raise ValueError("Candidate batching requires valid worlds and policy training steps")
    if config.training_episode_horizon < 1:
        raise ValueError("training_episode_horizon must be at least 1")
    if config.verified_evaluator not in {"mjwarp", "gym"}:
        raise ValueError("verified_evaluator must be one of: mjwarp, gym")
    validate_verification_audit_config(config)
    validate_reward_backend(config.reward_backend)
    validate_ppo_init_config(config)

    try:
        import mujoco_warp as mjw
        import warp as wp
    except ImportError as exc:
        raise RuntimeError(
            "The MJWarp evaluator requires optional GPU dependencies. Install with "
            "`pip install -e '.[mjwarp]'`."
        ) from exc

    started_at = time.monotonic()
    adapter = get_adapter(config.task)
    reward_programs = [
        VectorizedRewardProgram(
            component_expressions=candidate.component_expressions,
            expression=candidate.expression,
            allowed_names=adapter.reward_variables,
        )
        for candidate in candidates
    ]
    env = gym.make(config.task)
    try:
        mjm = env.unwrapped.model
        frame_skip = int(env.unwrapped.frame_skip)
        dt = float(mjm.opt.timestep * frame_skip)
        action_dim = int(mjm.nu)
        obs_dim = int(mjm.nq - 2 + mjm.nv)
        total_worlds = len(candidates) * config.worlds_per_candidate
        wp.init()
        with wp.ScopedDevice(config.device):
            model = mjw.put_model(mjm)
            data = mjw.make_data(mjm, nworld=total_worlds)
            wp.synchronize()
            policy, best_returns, summaries, graph_enabled = train_ppo_policy_gpu_batch(
                model=model,
                data=data,
                obs_dim=obs_dim,
                action_dim=action_dim,
                reward_programs=reward_programs,
                config=config,
                dt=dt,
                frame_skip=frame_skip,
                wp=wp,
                mjw=mjw,
            )
    finally:
        env.close()

    audit_returns_by_candidate: list[list[float]] | None = None
    if config.verified_evaluator == "mjwarp":
        with wp.ScopedDevice(config.device):
            eval_total_worlds = len(candidates) * config.eval_episodes
            eval_data = mjw.make_data(mjm, nworld=eval_total_worlds)
            wp.synchronize()
            eval_returns_by_candidate = evaluate_batched_policy_in_mjwarp(
                task=config.task,
                model=model,
                data=eval_data,
                policy=policy,
                candidate_count=len(candidates),
                action_dim=action_dim,
                eval_episodes=config.eval_episodes,
                episode_steps=config.verification_steps,
                seed=config.seed + 10_000,
                device=config.device,
                dt=dt,
                frame_skip=frame_skip,
                use_cuda_graph=config.use_cuda_graph,
                wp=wp,
                mjw=mjw,
            )
        if config.verified_audit_gym:
            audit_returns_by_candidate = [
                evaluate_policy_in_gym(
                    task=config.task,
                    policy=("batched_ppo", (policy, candidate_index)),
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    eval_episodes=config.eval_episodes,
                    seed=config.seed + 10_000,
                )
                for candidate_index in range(len(candidates))
            ]
    else:
        eval_returns_by_candidate = []
        for candidate_index in range(len(candidates)):
            eval_policy = ("batched_ppo", (policy, candidate_index))
            eval_returns = evaluate_policy_in_gym(
                task=config.task,
                policy=eval_policy,
                obs_dim=obs_dim,
                action_dim=action_dim,
                eval_episodes=config.eval_episodes,
                seed=config.seed + 10_000,
            )
            eval_returns_by_candidate.append(eval_returns)

    elapsed_seconds = time.monotonic() - started_at
    return [
        {
            "mean_reward": float(np.mean(eval_returns)),
            "std_reward": float(np.std(eval_returns)),
            "episode_rewards": eval_returns,
            "elapsed_seconds": elapsed_seconds,
            "metadata": {
                "sim_backend": "mjwarp",
                "mjwarp_evaluator": config.evaluator,
                "worlds_per_candidate": config.worlds_per_candidate,
                "total_batched_worlds": len(candidates) * config.worlds_per_candidate,
                "candidate_batching": True,
                "candidate_batch_size": len(candidates),
                "episode_steps": config.episode_steps,
                "training_episode_horizon": config.training_episode_horizon,
                "policy_iterations": config.policy_iterations,
                "training_world_steps": config.worlds_per_candidate
                * config.episode_steps
                * config.policy_iterations,
                "ppo_horizon": config.ppo_horizon,
                "ppo_epochs": config.ppo_epochs,
                "ppo_minibatch_size": config.ppo_minibatch_size,
                "ppo_hidden_sizes": [256, 128, 64],
                **base_policy_metadata(config),
                "rollout_mode": config.rollout_mode,
                "verified_evaluator": config.verified_evaluator,
                "verified_audit": build_verification_audit(
                    eval_returns,
                    None if audit_returns_by_candidate is None else audit_returns_by_candidate[index],
                    config,
                ),
                "reward_backend": config.reward_backend,
                "cuda_graph_requested": config.use_cuda_graph,
                "cuda_graph_enabled": graph_enabled,
                "frame_skip": frame_skip,
                "evaluation_episode_steps": config.verification_steps,
                "reward_components": reward_programs[index].component_expressions,
                "best_shaped_return": best_returns[index],
                "iteration_summaries": summaries[index],
            },
        }
        for index, eval_returns in enumerate(eval_returns_by_candidate)
    ]


def pretrain_original_reward_policy(config: MjwarpEvaluatorConfig, output_path: str | Path) -> dict[str, Any]:
    if config.task != "Ant-v5":
        raise ValueError("The MJWarp base-policy pretrainer currently supports only Ant-v5")
    if config.evaluator != "ppo":
        raise ValueError("Base-policy pretraining requires evaluator='ppo'")
    validate_ppo_init_config(config)
    validate_reward_backend(config.reward_backend)
    validate_verification_audit_config(config)

    try:
        import mujoco_warp as mjw
        import torch
        import warp as wp
    except ImportError as exc:
        raise RuntimeError(
            "The MJWarp base-policy pretrainer requires optional GPU dependencies. Install with "
            "`pip install -e '.[mjwarp]'`."
        ) from exc

    started_at = time.monotonic()
    reward_program = VectorizedRewardProgram(
        component_expressions={
            "forward": "forward_reward",
            "survive": "survive_reward",
            "control": "-control_cost",
        },
        expression="",
        allowed_names=get_adapter(config.task).reward_variables,
    )
    env = gym.make(config.task)
    try:
        mjm = env.unwrapped.model
        frame_skip = int(env.unwrapped.frame_skip)
        dt = float(mjm.opt.timestep * frame_skip)
        action_dim = int(mjm.nu)
        obs_dim = int(mjm.nq - 2 + mjm.nv)
        wp.init()
        with wp.ScopedDevice(config.device):
            model = mjw.put_model(mjm)
            data = mjw.make_data(mjm, nworld=config.worlds_per_candidate)
            wp.synchronize()
            policy, best_shaped_return, iteration_summaries = train_ppo_policy(
                model=model,
                data=data,
                mjm=mjm,
                obs_dim=obs_dim,
                action_dim=action_dim,
                reward_program=reward_program,
                config=config,
                dt=dt,
                frame_skip=frame_skip,
                wp=wp,
                mjw=mjw,
            )
            eval_data = mjw.make_data(mjm, nworld=config.eval_episodes)
            wp.synchronize()
            eval_returns = evaluate_policy_in_mjwarp(
                task=config.task,
                model=model,
                data=eval_data,
                mjm=mjm,
                policy=("ppo", policy),
                obs_dim=obs_dim,
                action_dim=action_dim,
                eval_episodes=config.eval_episodes,
                episode_steps=config.verification_steps,
                seed=config.seed + 10_000,
                device=config.device,
                dt=dt,
                frame_skip=frame_skip,
                use_cuda_graph=config.use_cuda_graph,
                wp=wp,
                mjw=mjw,
            )
    finally:
        env.close()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_state_dict": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
        "metadata": {
            "task": config.task,
            "reward": "original_mjwarp_ant_reward",
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "worlds_per_candidate": config.worlds_per_candidate,
            "episode_steps": config.episode_steps,
            "policy_iterations": config.policy_iterations,
            "training_world_steps": config.worlds_per_candidate * config.episode_steps * config.policy_iterations,
            "ppo_horizon": config.ppo_horizon,
            "ppo_epochs": config.ppo_epochs,
            "ppo_minibatch_size": config.ppo_minibatch_size,
            "ppo_learning_rate": config.ppo_learning_rate,
            "verified_evaluator": "mjwarp",
            "verification_steps": config.verification_steps,
            "eval_episodes": config.eval_episodes,
            "seed": config.seed,
            "device": config.device,
            "best_shaped_return": best_shaped_return,
            "mean_verified_return": float(np.mean(eval_returns)),
            "std_verified_return": float(np.std(eval_returns)),
            "episode_rewards": eval_returns,
            "elapsed_seconds": time.monotonic() - started_at,
            "config": asdict(config),
            "iteration_summaries": iteration_summaries,
        },
    }
    torch.save(payload, output)
    return {
        "output": output.as_posix(),
        "mean_verified_return": payload["metadata"]["mean_verified_return"],
        "std_verified_return": payload["metadata"]["std_verified_return"],
        "episode_rewards": eval_returns,
        "elapsed_seconds": payload["metadata"]["elapsed_seconds"],
    }


def dataclass_replace(config: MjwarpEvaluatorConfig, **updates: Any) -> MjwarpEvaluatorConfig:
    values = asdict(config)
    values.update(updates)
    return MjwarpEvaluatorConfig(**values)


def rollout_policy_population(
    *,
    model: Any,
    data: Any,
    mjm: Any,
    params: np.ndarray,
    reward_program: "VectorizedRewardProgram",
    episode_steps: int,
    dt: float,
    frame_skip: int,
    device: str,
    wp: Any,
    mjw: Any,
) -> tuple[np.ndarray, np.ndarray]:
    worlds, param_dim = params.shape
    action_dim = int(mjm.nu)
    obs_dim = int(param_dim // action_dim - 1)
    weights = params[:, : obs_dim * action_dim].reshape(worlds, obs_dim, action_dim)
    biases = params[:, obs_dim * action_dim :]
    shaped_returns = np.zeros(worlds, dtype=np.float32)
    true_returns = np.zeros(worlds, dtype=np.float32)
    terminated = np.zeros(worlds, dtype=bool)
    component_tracker = RewardComponentTracker(reward_program.component_names)

    mjw.reset_data(model, data)
    wp.synchronize()
    for _step in range(episode_steps):
        qpos_before = np.asarray(data.qpos.numpy(), dtype=np.float32)
        qvel_before = np.asarray(data.qvel.numpy(), dtype=np.float32)
        obs = ant_policy_obs(qpos_before, qvel_before)
        action = np.tanh(np.einsum("wo,woa->wa", obs, weights) + biases).astype(np.float32)
        wp.copy(data.ctrl, wp.array(action, dtype=wp.float32, device=device))
        mjwarp_control_step(model, data, mjw=mjw, frame_skip=frame_skip)
        wp.synchronize()

        qpos_after = np.asarray(data.qpos.numpy(), dtype=np.float32)
        qvel_after = np.asarray(data.qvel.numpy(), dtype=np.float32)
        x_velocity = (qpos_after[:, 0] - qpos_before[:, 0]) / dt
        y_velocity = (qpos_after[:, 1] - qpos_before[:, 1]) / dt
        torso_z = qpos_after[:, 2]
        healthy = np.isfinite(qpos_after).all(axis=1) & np.isfinite(qvel_after).all(axis=1)
        healthy &= (torso_z >= 0.2) & (torso_z <= 1.0)
        just_terminated = ~terminated & ~healthy
        active = ~terminated
        terminated |= just_terminated

        action_l2 = np.square(action).sum(axis=1)
        forward_reward = x_velocity
        control_cost = 0.5 * 1.0e-2 * action_l2
        survive_reward = np.where(healthy, 1.0, 0.0)
        original_reward = forward_reward + survive_reward - control_cost
        context = {
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
            "control_cost": control_cost,
            "survive_reward": survive_reward,
            "torso_z": torso_z,
            "action_l2": action_l2,
            "original_reward": original_reward,
            "healthy": healthy,
            "terminated": just_terminated,
            "truncated": np.zeros(worlds, dtype=bool),
        }
        component_values = reward_program.components(context)
        component_tracker.update(component_values)
        shaped_reward = reward_program.total_from_components(component_values)
        shaped_returns += np.where(active, shaped_reward, 0.0).astype(np.float32)
        true_returns += np.where(active, original_reward, 0.0).astype(np.float32)
        if terminated.all():
            break
    return shaped_returns, true_returns, component_tracker.summary()


def train_search_policy(
    *,
    model: Any,
    data: Any,
    mjm: Any,
    obs_dim: int,
    action_dim: int,
    reward_program: "VectorizedRewardProgram",
    config: MjwarpEvaluatorConfig,
    dt: float,
    frame_skip: int,
    rng: np.random.Generator,
    wp: Any,
    mjw: Any,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    param_dim = obs_dim * action_dim + action_dim
    mean = np.zeros(param_dim, dtype=np.float32)
    std = np.full(param_dim, config.init_std, dtype=np.float32)
    best_params = mean.copy()
    best_shaped_return = float("-inf")
    iteration_summaries = []
    for iteration in range(config.policy_iterations):
        params = rng.normal(mean, std, size=(config.worlds_per_candidate, param_dim)).astype(np.float32)
        shaped_returns, true_returns, component_stats = rollout_policy_population(
            model=model,
            data=data,
            mjm=mjm,
            params=params,
            reward_program=reward_program,
            episode_steps=config.episode_steps,
            dt=dt,
            frame_skip=frame_skip,
            device=config.device,
            wp=wp,
            mjw=mjw,
        )
        elite_count = max(1, int(config.worlds_per_candidate * config.elite_frac))
        elite_indices = np.argpartition(shaped_returns, -elite_count)[-elite_count:]
        elite_params = params[elite_indices]
        mean = elite_params.mean(axis=0).astype(np.float32)
        std = np.maximum(elite_params.std(axis=0).astype(np.float32), config.min_std)
        best_index = int(np.argmax(shaped_returns))
        if float(shaped_returns[best_index]) > best_shaped_return:
            best_shaped_return = float(shaped_returns[best_index])
            best_params = params[best_index].copy()
        iteration_summaries.append(
            {
                "iteration": iteration,
                "mean_shaped_return": float(np.mean(shaped_returns)),
                "best_shaped_return": float(np.max(shaped_returns)),
                "best_true_return_in_population": float(true_returns[best_index]),
                "reward_component_stats": component_stats,
            }
        )
    return best_params, best_shaped_return, iteration_summaries


def train_ppo_policy(
    *,
    model: Any,
    data: Any,
    mjm: Any,
    obs_dim: int,
    action_dim: int,
    reward_program: "VectorizedRewardProgram",
    config: MjwarpEvaluatorConfig,
    dt: float,
    frame_skip: int,
    wp: Any,
    mjw: Any,
) -> tuple[Any, float, list[dict[str, float]]]:
    seed_torch_policy_rng(config.seed)
    if config.rollout_mode == "gpu":
        return train_ppo_policy_gpu(
            model=model,
            data=data,
            obs_dim=obs_dim,
            action_dim=action_dim,
            reward_program=reward_program,
            config=config,
            dt=dt,
            frame_skip=frame_skip,
            wp=wp,
            mjw=mjw,
        )
    return train_ppo_policy_host(
        model=model,
        data=data,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_program=reward_program,
        config=config,
        dt=dt,
        frame_skip=frame_skip,
        wp=wp,
        mjw=mjw,
    )


def seed_torch_policy_rng(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_policy_std(policy: Any, init_std: float) -> None:
    import torch

    if init_std <= 0.0:
        raise ValueError("init_std must be positive")
    with torch.no_grad():
        policy.log_std.fill_(math.log(init_std))


def load_base_policy_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Base policy checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Base policy checkpoint must be a dict: {checkpoint_path}")
    state = checkpoint.get("policy_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Base policy checkpoint is missing policy_state_dict: {checkpoint_path}")
    return checkpoint


def load_base_policy_into_single(
    policy: Any,
    checkpoint_path: str | Path,
    *,
    obs_dim: int,
    action_dim: int,
    torch_device: Any,
) -> dict[str, Any]:
    checkpoint = load_base_policy_checkpoint(checkpoint_path, map_location=torch_device)
    _validate_base_policy_dimensions(checkpoint, obs_dim=obs_dim, action_dim=action_dim, path=checkpoint_path)
    policy.load_state_dict(checkpoint.get("policy_state_dict", checkpoint))
    return checkpoint


def load_base_policy_into_batched(
    policy: Any,
    checkpoint_path: str | Path,
    *,
    candidate_count: int,
    obs_dim: int,
    action_dim: int,
    torch_device: Any,
) -> dict[str, Any]:
    import torch

    checkpoint = load_base_policy_checkpoint(checkpoint_path, map_location=torch_device)
    _validate_base_policy_dimensions(checkpoint, obs_dim=obs_dim, action_dim=action_dim, path=checkpoint_path)
    base = AntActorCritic(obs_dim, action_dim).to(torch_device)
    base.load_state_dict(checkpoint.get("policy_state_dict", checkpoint))
    linear_layers = [base.backbone[0], base.backbone[2], base.backbone[4], base.actor_mean, base.critic]
    with torch.no_grad():
        for index, layer in enumerate(linear_layers):
            getattr(policy, f"weight_{index}").copy_(layer.weight.unsqueeze(0).repeat(candidate_count, 1, 1))
            getattr(policy, f"bias_{index}").copy_(layer.bias.unsqueeze(0).repeat(candidate_count, 1))
        policy.log_std.copy_(base.log_std.unsqueeze(0).repeat(candidate_count, 1))
    return checkpoint


def maybe_initialize_policy_from_base(
    policy: Any,
    config: MjwarpEvaluatorConfig,
    *,
    candidate_count: int | None,
    obs_dim: int,
    action_dim: int,
    torch_device: Any,
) -> dict[str, Any] | None:
    if config.ppo_init_mode == "scratch":
        return None
    if not config.base_policy_checkpoint:
        raise ValueError("base_policy_checkpoint is required when ppo_init_mode='base'")
    if candidate_count is None:
        return load_base_policy_into_single(
            policy,
            config.base_policy_checkpoint,
            obs_dim=obs_dim,
            action_dim=action_dim,
            torch_device=torch_device,
        )
    return load_base_policy_into_batched(
        policy,
        config.base_policy_checkpoint,
        candidate_count=candidate_count,
        obs_dim=obs_dim,
        action_dim=action_dim,
        torch_device=torch_device,
    )


def _validate_base_policy_dimensions(
    checkpoint: dict[str, Any],
    *,
    obs_dim: int,
    action_dim: int,
    path: str | Path,
) -> None:
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    checkpoint_obs_dim = metadata.get("obs_dim")
    checkpoint_action_dim = metadata.get("action_dim")
    if checkpoint_obs_dim is not None and int(checkpoint_obs_dim) != obs_dim:
        raise ValueError(f"Base policy checkpoint {path} obs_dim={checkpoint_obs_dim} does not match {obs_dim}")
    if checkpoint_action_dim is not None and int(checkpoint_action_dim) != action_dim:
        raise ValueError(
            f"Base policy checkpoint {path} action_dim={checkpoint_action_dim} does not match {action_dim}"
        )


def train_ppo_policy_host(
    *,
    model: Any,
    data: Any,
    obs_dim: int,
    action_dim: int,
    reward_program: "VectorizedRewardProgram",
    config: MjwarpEvaluatorConfig,
    dt: float,
    frame_skip: int,
    wp: Any,
    mjw: Any,
) -> tuple[Any, float, list[dict[str, float]]]:
    import torch

    torch_device = torch.device("cuda" if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    policy = AntActorCritic(obs_dim, action_dim).to(torch_device)
    initialize_policy_std(policy, config.init_std)
    maybe_initialize_policy_from_base(
        policy,
        config,
        candidate_count=None,
        obs_dim=obs_dim,
        action_dim=action_dim,
        torch_device=torch_device,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo_learning_rate)
    iteration_summaries = []
    best_shaped_return = float("-inf")
    best_true_return = float("-inf")

    mjw.reset_data(model, data)
    wp.synchronize()
    dones = np.zeros(config.worlds_per_candidate, dtype=bool)
    episode_shaped = np.zeros(config.worlds_per_candidate, dtype=np.float32)
    episode_true = np.zeros(config.worlds_per_candidate, dtype=np.float32)
    component_tracker = RewardComponentTracker(reward_program.component_names)

    chunks_per_iteration = max(1, int(np.ceil(config.episode_steps / config.ppo_horizon)))
    update_index = 0
    for iteration in range(config.policy_iterations):
        for chunk in range(chunks_per_iteration):
            rollout_steps = min(config.ppo_horizon, config.episode_steps - chunk * config.ppo_horizon)
            batch_obs = []
            batch_actions = []
            batch_logprobs = []
            batch_values = []
            batch_rewards = []
            batch_dones = []
            for _step in range(rollout_steps):
                qpos_before = np.asarray(data.qpos.numpy(), dtype=np.float32)
                qvel_before = np.asarray(data.qvel.numpy(), dtype=np.float32)
                obs_np = ant_policy_obs(qpos_before, qvel_before)
                obs = torch.as_tensor(obs_np, dtype=torch.float32, device=torch_device)
                with torch.no_grad():
                    action_t, logprob_t, _, value_t = policy.act(obs)
                action = action_t.detach().cpu().numpy().astype(np.float32)
                wp.copy(data.ctrl, wp.array(action, dtype=wp.float32, device=config.device))
                mjwarp_control_step(model, data, mjw=mjw, frame_skip=frame_skip)
                wp.synchronize()

                qpos_after = np.asarray(data.qpos.numpy(), dtype=np.float32)
                qvel_after = np.asarray(data.qvel.numpy(), dtype=np.float32)
                shaped_reward, true_reward, next_dones, component_values = ant_rewards(
                    qpos_before=qpos_before,
                    qpos_after=qpos_after,
                    qvel_after=qvel_after,
                    action=action,
                    reward_program=reward_program,
                    dt=dt,
                )
                component_tracker.update(component_values)
                active = ~dones
                shaped_reward = np.where(active, shaped_reward, 0.0).astype(np.float32)
                true_reward = np.where(active, true_reward, 0.0).astype(np.float32)
                episode_shaped += shaped_reward
                episode_true += true_reward
                dones |= next_dones

                batch_obs.append(obs_np)
                batch_actions.append(action)
                batch_logprobs.append(logprob_t.detach().cpu().numpy().astype(np.float32))
                batch_values.append(value_t.detach().cpu().numpy().astype(np.float32))
                batch_rewards.append(shaped_reward)
                batch_dones.append(dones.astype(np.float32))

            qpos_before = np.asarray(data.qpos.numpy(), dtype=np.float32)
            qvel = np.asarray(data.qvel.numpy(), dtype=np.float32)
            next_obs = torch.as_tensor(ant_policy_obs(qpos_before, qvel), dtype=torch.float32, device=torch_device)
            with torch.no_grad():
                next_value = policy.value(next_obs).detach().cpu().numpy().astype(np.float32)

            rewards = np.asarray(batch_rewards, dtype=np.float32)
            values = np.asarray(batch_values, dtype=np.float32)
            done_flags = np.asarray(batch_dones, dtype=np.float32)
            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                dones=done_flags,
                next_value=next_value,
                gamma=config.ppo_gamma,
                gae_lambda=config.ppo_gae_lambda,
            )
            flat_obs = torch.as_tensor(np.asarray(batch_obs, dtype=np.float32).reshape(-1, obs_dim), device=torch_device)
            flat_actions = torch.as_tensor(np.asarray(batch_actions, dtype=np.float32).reshape(-1, action_dim), device=torch_device)
            flat_old_logprobs = torch.as_tensor(np.asarray(batch_logprobs, dtype=np.float32).reshape(-1), device=torch_device)
            flat_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=torch_device)
            flat_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=torch_device)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / flat_advantages.std().clamp_min(1e-6)

            ppo_update(
                policy=policy,
                optimizer=optimizer,
                obs=flat_obs,
                actions=flat_actions,
                old_logprobs=flat_old_logprobs,
                advantages=flat_advantages,
                returns=flat_returns,
                config=config,
            )
            mean_shaped = float(np.mean(episode_shaped))
            max_shaped = float(np.max(episode_shaped))
            best_shaped_return = max(best_shaped_return, max_shaped)
            best_true_return = max(best_true_return, float(np.max(episode_true)))
            iteration_summaries.append(
                {
                    "iteration": iteration,
                    "ppo_update": update_index,
                    "mean_shaped_return": mean_shaped,
                    "best_shaped_return": max_shaped,
                    "mean_true_return_in_population": float(np.mean(episode_true)),
                    "best_true_return_in_population": best_true_return,
                    "reward_component_stats": component_tracker.summary(),
                }
            )
            update_index += 1
            if dones.all():
                episode_shaped.fill(0.0)
                episode_true.fill(0.0)
                dones.fill(False)
                mjw.reset_data(model, data)
                wp.synchronize()

    policy.eval()
    return policy, best_shaped_return, iteration_summaries


def train_ppo_policy_gpu(
    *,
    model: Any,
    data: Any,
    obs_dim: int,
    action_dim: int,
    reward_program: "VectorizedRewardProgram",
    config: MjwarpEvaluatorConfig,
    dt: float,
    frame_skip: int,
    wp: Any,
    mjw: Any,
) -> tuple[Any, float, list[dict[str, float]]]:
    import torch

    torch_device = torch.device("cuda" if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    policy = AntActorCritic(obs_dim, action_dim).to(torch_device)
    initialize_policy_std(policy, config.init_std)
    maybe_initialize_policy_from_base(
        policy,
        config,
        candidate_count=None,
        obs_dim=obs_dim,
        action_dim=action_dim,
        torch_device=torch_device,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo_learning_rate)
    torch_reward_program = make_torch_reward_program(
        component_expressions=reward_program.component_expressions,
        expression=reward_program.expression,
        allowed_names=get_adapter(config.task).reward_variables,
        backend=config.reward_backend,
    )
    iteration_summaries: list[dict[str, float]] = []
    best_shaped_return = float("-inf")

    pending_metrics: list[dict[str, Any]] = []
    with torch_warp_stream(wp, torch, torch_device):
        mjw.reset_data(model, data)
        qpos = wp.to_torch(data.qpos)
        qvel = wp.to_torch(data.qvel)
        ctrl = wp.to_torch(data.ctrl)
        physics_graph = create_control_step_graph(
            model=model,
            data=data,
            mjw=mjw,
            wp=wp,
            frame_skip=frame_skip,
            enabled=config.use_cuda_graph,
            device=config.device,
        )
        episode_lengths = torch.zeros(config.worlds_per_candidate, dtype=torch.int32, device=torch_device)
        episode_shaped = torch.zeros(config.worlds_per_candidate, dtype=torch.float32, device=torch_device)
        episode_true = torch.zeros(config.worlds_per_candidate, dtype=torch.float32, device=torch_device)
        component_tracker = TorchRewardComponentTracker(torch_reward_program.component_names, torch_device)
        best_shaped_t = torch.full((), float("-inf"), dtype=torch.float32, device=torch_device)
        best_true_t = torch.full((), float("-inf"), dtype=torch.float32, device=torch_device)

        chunks_per_iteration = max(1, int(np.ceil(config.episode_steps / config.ppo_horizon)))
        update_index = 0
        for iteration in range(config.policy_iterations):
            for chunk in range(chunks_per_iteration):
                rollout_steps = min(config.ppo_horizon, config.episode_steps - chunk * config.ppo_horizon)
                batch_obs = torch.empty(
                    (rollout_steps, config.worlds_per_candidate, obs_dim), dtype=torch.float32, device=torch_device
                )
                batch_actions = torch.empty(
                    (rollout_steps, config.worlds_per_candidate, action_dim), dtype=torch.float32, device=torch_device
                )
                batch_logprobs = torch.empty(
                    (rollout_steps, config.worlds_per_candidate), dtype=torch.float32, device=torch_device
                )
                batch_values = torch.empty_like(batch_logprobs)
                batch_rewards = torch.empty_like(batch_logprobs)
                batch_dones = torch.empty_like(batch_logprobs)

                for step in range(rollout_steps):
                    obs = ant_policy_obs_torch(qpos, qvel)
                    qpos_xy_before = qpos[:, :2].clone()
                    with torch.no_grad():
                        action, logprob, _, value = policy.act(obs)
                    ctrl.copy_(action)
                    advance_control_step(
                        model=model,
                        data=data,
                        mjw=mjw,
                        wp=wp,
                        frame_skip=frame_skip,
                        graph=physics_graph,
                    )
                    shaped_reward, true_reward, next_dones, component_values = ant_rewards_torch(
                        qpos_xy_before=qpos_xy_before,
                        qpos_after=qpos,
                        qvel_after=qvel,
                        action=action,
                        reward_program=torch_reward_program,
                        dt=dt,
                    )
                    component_tracker.update(component_values)
                    episode_shaped.add_(shaped_reward)
                    episode_true.add_(true_reward)
                    episode_lengths.add_(1)
                    reset_mask = next_dones | (episode_lengths >= config.training_episode_horizon)
                    best_shaped_t = torch.maximum(best_shaped_t, episode_shaped.max())
                    best_true_t = torch.maximum(best_true_t, episode_true.max())

                    batch_obs[step].copy_(obs)
                    batch_actions[step].copy_(action)
                    batch_logprobs[step].copy_(logprob)
                    batch_values[step].copy_(value)
                    batch_rewards[step].copy_(shaped_reward)
                    batch_dones[step].copy_(reset_mask)
                    reset_mjwarp_worlds(model, data, reset_mask, mjw=mjw, wp=wp)
                    episode_shaped.masked_fill_(reset_mask, 0.0)
                    episode_true.masked_fill_(reset_mask, 0.0)
                    episode_lengths.masked_fill_(reset_mask, 0)

                with torch.no_grad():
                    next_value = policy.value(ant_policy_obs_torch(qpos, qvel))
                advantages, returns = compute_gae_torch(
                    rewards=batch_rewards,
                    values=batch_values,
                    dones=batch_dones,
                    next_value=next_value,
                    gamma=config.ppo_gamma,
                    gae_lambda=config.ppo_gae_lambda,
                )
                flat_obs = batch_obs.reshape(-1, obs_dim)
                flat_actions = batch_actions.reshape(-1, action_dim)
                flat_old_logprobs = batch_logprobs.reshape(-1)
                flat_advantages = advantages.reshape(-1)
                flat_returns = returns.reshape(-1)
                flat_advantages = (flat_advantages - flat_advantages.mean()) / flat_advantages.std().clamp_min(1e-6)

                ppo_update(
                    policy=policy,
                    optimizer=optimizer,
                    obs=flat_obs,
                    actions=flat_actions,
                    old_logprobs=flat_old_logprobs,
                    advantages=flat_advantages,
                    returns=flat_returns,
                    config=config,
                )
                max_shaped_t = episode_shaped.max()
                best_shaped_t = torch.maximum(best_shaped_t, max_shaped_t)
                pending_metrics.append(
                    {
                        "iteration": iteration,
                        "ppo_update": update_index,
                        "mean_shaped_t": episode_shaped.mean(),
                        "max_shaped_t": max_shaped_t,
                        "mean_true_t": episode_true.mean(),
                        "best_true_t": best_true_t.clone(),
                        "component_snapshot": component_tracker.snapshot(),
                    }
                )
                update_index += 1
    iteration_summaries = _materialize_metric_log(pending_metrics)
    best_shaped_return = float(best_shaped_t.item())
    policy.eval()
    return policy, best_shaped_return, iteration_summaries


def train_ppo_policy_gpu_batch(
    *,
    model: Any,
    data: Any,
    obs_dim: int,
    action_dim: int,
    reward_programs: list["VectorizedRewardProgram"],
    config: MjwarpEvaluatorConfig,
    dt: float,
    frame_skip: int,
    wp: Any,
    mjw: Any,
) -> tuple[Any, list[float], list[list[dict[str, Any]]], bool]:
    import torch

    candidate_count = len(reward_programs)
    worlds = config.worlds_per_candidate
    torch_device = torch.device("cuda" if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    seed_torch_policy_rng(config.seed)
    policy = BatchedAntActorCritic(candidate_count, obs_dim, action_dim).to(torch_device)
    initialize_policy_std(policy, config.init_std)
    maybe_initialize_policy_from_base(
        policy,
        config,
        candidate_count=candidate_count,
        obs_dim=obs_dim,
        action_dim=action_dim,
        torch_device=torch_device,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo_learning_rate)
    allowed_names = get_adapter(config.task).reward_variables
    torch_programs = [
        make_torch_reward_program(
            component_expressions=program.component_expressions,
            expression=program.expression,
            allowed_names=allowed_names,
            backend=config.reward_backend,
        )
        for program in reward_programs
    ]
    pending_metrics: list[dict[str, Any]] = []

    with torch_warp_stream(wp, torch, torch_device):
        mjw.reset_data(model, data)
        qpos = wp.to_torch(data.qpos).reshape(candidate_count, worlds, -1)
        qvel = wp.to_torch(data.qvel).reshape(candidate_count, worlds, -1)
        ctrl = wp.to_torch(data.ctrl).reshape(candidate_count, worlds, action_dim)
        physics_graph = create_control_step_graph(
            model=model,
            data=data,
            mjw=mjw,
            wp=wp,
            frame_skip=frame_skip,
            enabled=config.use_cuda_graph,
            device=config.device,
        )
        episode_lengths = torch.zeros((candidate_count, worlds), dtype=torch.int32, device=torch_device)
        episode_shaped = torch.zeros((candidate_count, worlds), dtype=torch.float32, device=torch_device)
        episode_true = torch.zeros_like(episode_shaped)
        component_trackers = [
            TorchRewardComponentTracker(program.component_names, torch_device) for program in torch_programs
        ]
        best_returns_t = torch.full((candidate_count,), float("-inf"), dtype=torch.float32, device=torch_device)
        best_true_t = torch.full((candidate_count,), float("-inf"), dtype=torch.float32, device=torch_device)

        chunks_per_iteration = max(1, int(np.ceil(config.episode_steps / config.ppo_horizon)))
        update_index = 0
        for iteration in range(config.policy_iterations):
            for chunk in range(chunks_per_iteration):
                rollout_steps = min(config.ppo_horizon, config.episode_steps - chunk * config.ppo_horizon)
                batch_obs = torch.empty(
                    (rollout_steps, candidate_count, worlds, obs_dim), dtype=torch.float32, device=torch_device
                )
                batch_actions = torch.empty(
                    (rollout_steps, candidate_count, worlds, action_dim), dtype=torch.float32, device=torch_device
                )
                batch_logprobs = torch.empty(
                    (rollout_steps, candidate_count, worlds), dtype=torch.float32, device=torch_device
                )
                batch_values = torch.empty_like(batch_logprobs)
                batch_rewards = torch.empty_like(batch_logprobs)
                batch_dones = torch.empty_like(batch_logprobs)

                for step in range(rollout_steps):
                    obs = ant_policy_obs_torch_batched(qpos, qvel)
                    qpos_xy_before = qpos[:, :, :2].clone()
                    shared_noise = torch.randn((1, worlds, action_dim), dtype=torch.float32, device=torch_device)
                    shared_noise = shared_noise.expand(candidate_count, -1, -1)
                    with torch.no_grad():
                        action, logprob, _, value = policy.act(obs, shared_noise)
                    ctrl.copy_(action)
                    advance_control_step(
                        model=model,
                        data=data,
                        mjw=mjw,
                        wp=wp,
                        frame_skip=frame_skip,
                        graph=physics_graph,
                    )
                    shaped_parts = []
                    true_parts = []
                    done_parts = []
                    for index, reward_program in enumerate(torch_programs):
                        shaped, true, next_dones, components = ant_rewards_torch(
                            qpos_xy_before=qpos_xy_before[index],
                            qpos_after=qpos[index],
                            qvel_after=qvel[index],
                            action=action[index],
                            reward_program=reward_program,
                            dt=dt,
                        )
                        component_trackers[index].update(components)
                        shaped_parts.append(shaped)
                        true_parts.append(true)
                        done_parts.append(next_dones)
                    shaped_reward = torch.stack(shaped_parts)
                    true_reward = torch.stack(true_parts)
                    next_dones = torch.stack(done_parts)
                    episode_shaped.add_(shaped_reward)
                    episode_true.add_(true_reward)
                    episode_lengths.add_(1)
                    reset_mask = next_dones | (episode_lengths >= config.training_episode_horizon)
                    best_returns_t = torch.maximum(best_returns_t, episode_shaped.max(dim=1).values)
                    best_true_t = torch.maximum(best_true_t, episode_true.max(dim=1).values)
                    batch_obs[step].copy_(obs)
                    batch_actions[step].copy_(action)
                    batch_logprobs[step].copy_(logprob)
                    batch_values[step].copy_(value)
                    batch_rewards[step].copy_(shaped_reward)
                    batch_dones[step].copy_(reset_mask)
                    reset_mjwarp_worlds(model, data, reset_mask, mjw=mjw, wp=wp)
                    episode_shaped.masked_fill_(reset_mask, 0.0)
                    episode_true.masked_fill_(reset_mask, 0.0)
                    episode_lengths.masked_fill_(reset_mask, 0)

                with torch.no_grad():
                    next_value = policy.value(ant_policy_obs_torch_batched(qpos, qvel))
                advantages, returns = compute_gae_torch(
                    rewards=batch_rewards,
                    values=batch_values,
                    dones=batch_dones,
                    next_value=next_value,
                    gamma=config.ppo_gamma,
                    gae_lambda=config.ppo_gae_lambda,
                )
                flat_obs = batch_obs.permute(1, 0, 2, 3).reshape(candidate_count, -1, obs_dim)
                flat_actions = batch_actions.permute(1, 0, 2, 3).reshape(candidate_count, -1, action_dim)
                flat_logprobs = batch_logprobs.permute(1, 0, 2).reshape(candidate_count, -1)
                flat_advantages = advantages.permute(1, 0, 2).reshape(candidate_count, -1)
                flat_returns = returns.permute(1, 0, 2).reshape(candidate_count, -1)
                means = flat_advantages.mean(dim=1, keepdim=True)
                stds = flat_advantages.std(dim=1, keepdim=True).clamp_min(1e-6)
                flat_advantages = (flat_advantages - means) / stds
                ppo_update_batched(
                    policy=policy,
                    optimizer=optimizer,
                    obs=flat_obs,
                    actions=flat_actions,
                    old_logprobs=flat_logprobs,
                    advantages=flat_advantages,
                    returns=flat_returns,
                    config=config,
                    candidate_count=candidate_count,
                )
                max_shaped = episode_shaped.max(dim=1).values
                best_returns_t = torch.maximum(best_returns_t, max_shaped)
                pending_metrics.append(
                    {
                        "iteration": iteration,
                        "ppo_update": update_index,
                        "mean_shaped_t": episode_shaped.mean(dim=1),
                        "max_shaped_t": max_shaped,
                        "mean_true_t": episode_true.mean(dim=1),
                        "best_true_t": best_true_t.clone(),
                        "component_snapshots": [tracker.snapshot() for tracker in component_trackers],
                    }
                )
                update_index += 1
    best_returns = best_returns_t.detach().cpu().tolist()
    summaries = _materialize_metric_log_batched(pending_metrics, candidate_count)
    policy.eval()
    return policy, best_returns, summaries, physics_graph is not None


def ant_rewards(
    *,
    qpos_before: np.ndarray,
    qpos_after: np.ndarray,
    qvel_after: np.ndarray,
    action: np.ndarray,
    reward_program: "VectorizedRewardProgram",
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x_velocity = (qpos_after[:, 0] - qpos_before[:, 0]) / dt
    y_velocity = (qpos_after[:, 1] - qpos_before[:, 1]) / dt
    torso_z = qpos_after[:, 2]
    healthy = np.isfinite(qpos_after).all(axis=1) & np.isfinite(qvel_after).all(axis=1)
    healthy &= (torso_z >= 0.2) & (torso_z <= 1.0)
    action_l2 = np.square(action).sum(axis=1)
    forward_reward = x_velocity
    control_cost = 0.5 * 1.0e-2 * action_l2
    survive_reward = np.where(healthy, 1.0, 0.0)
    original_reward = forward_reward + survive_reward - control_cost
    context = {
        "x_velocity": x_velocity,
        "y_velocity": y_velocity,
        "forward_reward": forward_reward,
        "control_cost": control_cost,
        "survive_reward": survive_reward,
        "torso_z": torso_z,
        "action_l2": action_l2,
        "original_reward": original_reward,
        "healthy": healthy,
        "terminated": ~healthy,
        "truncated": np.zeros(len(action), dtype=bool),
    }
    component_values = reward_program.components(context)
    shaped_reward = reward_program.total_from_components(component_values)
    return shaped_reward.astype(np.float32), original_reward.astype(np.float32), (~healthy), component_values


def ant_rewards_torch(
    *,
    qpos_xy_before: Any,
    qpos_after: Any,
    qvel_after: Any,
    action: Any,
    reward_program: "TorchRewardProgram",
    dt: float,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    x_velocity = (qpos_after[:, 0] - qpos_xy_before[:, 0]) / dt
    y_velocity = (qpos_after[:, 1] - qpos_xy_before[:, 1]) / dt
    torso_z = qpos_after[:, 2]
    healthy = torch.isfinite(qpos_after).all(dim=1) & torch.isfinite(qvel_after).all(dim=1)
    healthy &= (torso_z >= 0.2) & (torso_z <= 1.0)
    action_l2 = action.square().sum(dim=1)
    forward_reward = x_velocity
    control_cost = 0.5 * 1.0e-2 * action_l2
    survive_reward = healthy.to(dtype=torch.float32)
    original_reward = forward_reward + survive_reward - control_cost
    context = {
        "x_velocity": x_velocity,
        "y_velocity": y_velocity,
        "forward_reward": forward_reward,
        "control_cost": control_cost,
        "survive_reward": survive_reward,
        "torso_z": torso_z,
        "action_l2": action_l2,
        "original_reward": original_reward,
        "healthy": healthy,
        "terminated": ~healthy,
        "truncated": torch.zeros_like(healthy),
    }
    component_values = reward_program.components(context)
    shaped_reward = reward_program.total_from_components(component_values)
    return shaped_reward, original_reward.to(dtype=torch.float32), ~healthy, component_values


def compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[step]
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def compute_gae_torch(
    *,
    rewards: Any,
    values: Any,
    dones: Any,
    next_value: Any,
    gamma: float,
    gae_lambda: float,
) -> tuple[Any, Any]:
    import torch

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[step].to(dtype=rewards.dtype)
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    return advantages, advantages + values


def ppo_update(
    *,
    policy: Any,
    optimizer: Any,
    obs: Any,
    actions: Any,
    old_logprobs: Any,
    advantages: Any,
    returns: Any,
    config: MjwarpEvaluatorConfig,
) -> None:
    import torch

    batch_size = obs.shape[0]
    minibatch_size = min(config.ppo_minibatch_size, batch_size)
    for _epoch in range(config.ppo_epochs):
        permutation = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            new_logprob, entropy, value = policy.evaluate_actions(obs[idx], actions[idx])
            logratio = new_logprob - old_logprobs[idx]
            ratio = logratio.exp()
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * advantages[idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[idx] - value).square().mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + config.ppo_value_coef * value_loss - config.ppo_entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.ppo_max_grad_norm)
            optimizer.step()


def ppo_update_batched(
    *,
    policy: Any,
    optimizer: Any,
    obs: Any,
    actions: Any,
    old_logprobs: Any,
    advantages: Any,
    returns: Any,
    config: MjwarpEvaluatorConfig,
    candidate_count: int,
) -> None:
    import torch

    batch_size = obs.shape[1]
    minibatch_size = min(config.ppo_minibatch_size, batch_size)
    for _epoch in range(config.ppo_epochs):
        permutation = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            new_logprob, entropy, value = policy.evaluate_actions(obs[:, idx], actions[:, idx])
            ratio = (new_logprob - old_logprobs[:, idx]).exp()
            unclipped = ratio * advantages[:, idx]
            clipped = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * advantages[:, idx]
            policy_loss = -torch.min(unclipped, clipped).mean(dim=1)
            value_loss = 0.5 * (returns[:, idx] - value).square().mean(dim=1)
            entropy_loss = entropy.mean(dim=1)
            loss = (policy_loss + config.ppo_value_coef * value_loss - config.ppo_entropy_coef * entropy_loss).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_per_candidate_(policy.parameters(), config.ppo_max_grad_norm, candidate_count)
            optimizer.step()


def clip_grad_norm_per_candidate_(
    parameters: Any, max_norm: float, candidate_count: int
) -> None:
    """Scale each candidate's gradients independently so one noisy candidate cannot
    suppress updates for the others sharing the batched policy parameters."""
    import torch

    grads = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if parameter.shape[0] != candidate_count:
            raise ValueError(
                "Per-candidate grad clipping expects every batched-policy parameter to "
                f"have first dim equal to candidate_count={candidate_count}; got shape "
                f"{tuple(parameter.shape)}"
            )
        grads.append(parameter.grad)
    if not grads:
        return
    flattened = torch.cat([g.reshape(candidate_count, -1) for g in grads], dim=1)
    norms = flattened.norm(dim=1)
    coef = (max_norm / (norms + 1.0e-6)).clamp(max=1.0)
    for grad in grads:
        tail = (1,) * (grad.dim() - 1)
        grad.mul_(coef.view(candidate_count, *tail))


def ant_policy_obs(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    return np.concatenate([qpos[:, 2:], qvel], axis=1).astype(np.float32)


def ant_policy_obs_torch(qpos: Any, qvel: Any) -> Any:
    import torch

    return torch.cat((qpos[:, 2:], qvel), dim=1)


def ant_policy_obs_torch_batched(qpos: Any, qvel: Any) -> Any:
    import torch

    return torch.cat((qpos[:, :, 2:], qvel), dim=2)


def mjwarp_control_step(model: Any, data: Any, *, mjw: Any, frame_skip: int) -> None:
    for _frame in range(frame_skip):
        mjw.step(model, data)


def create_control_step_graph(
    *,
    model: Any,
    data: Any,
    mjw: Any,
    wp: Any,
    frame_skip: int,
    enabled: bool,
    device: str,
) -> Any | None:
    if not enabled or not device.startswith("cuda"):
        return None
    try:
        mjwarp_control_step(model, data, mjw=mjw, frame_skip=frame_skip)
        mjw.reset_data(model, data)
        wp.synchronize()
        with wp.ScopedCapture(device=device) as capture:
            mjwarp_control_step(model, data, mjw=mjw, frame_skip=frame_skip)
        mjw.reset_data(model, data)
        wp.synchronize()
        return capture.graph
    except Exception:
        mjw.reset_data(model, data)
        wp.synchronize()
        return None


def advance_control_step(
    *,
    model: Any,
    data: Any,
    mjw: Any,
    wp: Any,
    frame_skip: int,
    graph: Any | None,
) -> None:
    if graph is None:
        mjwarp_control_step(model, data, mjw=mjw, frame_skip=frame_skip)
    else:
        wp.capture_launch(graph)


def reset_mjwarp_worlds(model: Any, data: Any, reset_mask: Any, *, mjw: Any, wp: Any) -> None:
    """Reset completed worlds without a host synchronization."""
    reset = wp.from_torch(reset_mask.reshape(-1).contiguous(), dtype=wp.bool)
    mjw.reset_data(model, data, reset=reset)


def torch_warp_stream(wp: Any, torch: Any, device: Any) -> Any:
    if device.type != "cuda":
        return nullcontext()
    warp_device = f"cuda:{0 if device.index is None else device.index}"
    stream = torch.cuda.ExternalStream(wp.get_stream(warp_device).cuda_stream, device=device)
    return torch.cuda.stream(stream)


def evaluate_policy_in_mjwarp(
    *,
    task: str,
    model: Any,
    data: Any,
    mjm: Any,
    policy: tuple[str, Any],
    obs_dim: int,
    action_dim: int,
    eval_episodes: int,
    episode_steps: int,
    seed: int,
    device: str,
    dt: float,
    frame_skip: int,
    use_cuda_graph: bool,
    wp: Any,
    mjw: Any,
) -> list[float]:
    import torch

    del mjm
    torch_device = torch.device("cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    qpos_initial, qvel_initial = ant_reset_states(task, eval_episodes, seed)
    policy_type, policy_data = policy
    if policy_type == "linear":
        weights = torch.as_tensor(
            policy_data[: obs_dim * action_dim], dtype=torch.float32, device=torch_device
        ).reshape(obs_dim, action_dim)
        biases = torch.as_tensor(policy_data[weights.numel() :], dtype=torch.float32, device=torch_device)
    elif policy_type == "batched_ppo":
        torch_policy, candidate_index = policy_data
    else:
        torch_policy = policy_data

    with torch_warp_stream(wp, torch, torch_device):
        mjw.reset_data(model, data)
        qpos = wp.to_torch(data.qpos)
        qvel = wp.to_torch(data.qvel)
        ctrl = wp.to_torch(data.ctrl)
        physics_graph = create_control_step_graph(
            model=model,
            data=data,
            mjw=mjw,
            wp=wp,
            frame_skip=frame_skip,
            enabled=use_cuda_graph,
            device=device,
        )
        qpos.copy_(torch.as_tensor(qpos_initial, dtype=torch.float32, device=torch_device))
        qvel.copy_(torch.as_tensor(qvel_initial, dtype=torch.float32, device=torch_device))
        returns = torch.zeros(eval_episodes, dtype=torch.float32, device=torch_device)
        dones = torch.zeros(eval_episodes, dtype=torch.bool, device=torch_device)
        for _step in range(episode_steps):
            obs = ant_policy_obs_torch(qpos, qvel)
            qpos_xy_before = qpos[:, :2].clone()
            with torch.no_grad():
                if policy_type == "linear":
                    action = torch.tanh(obs @ weights + biases)
                elif policy_type == "batched_ppo":
                    action = torch_policy.mean_action_for_candidate(obs, candidate_index)
                else:
                    action = torch_policy.mean_action(obs)
            ctrl.copy_(action)
            advance_control_step(model=model, data=data, mjw=mjw, wp=wp, frame_skip=frame_skip, graph=physics_graph)
            true_reward, next_dones = original_ant_reward_torch(qpos_xy_before, qpos, qvel, action, dt)
            returns.add_(torch.where(~dones, true_reward, torch.zeros_like(true_reward)))
            dones.logical_or_(next_dones)
        return returns.detach().cpu().tolist()


def evaluate_batched_policy_in_mjwarp(
    *,
    task: str,
    model: Any,
    data: Any,
    policy: Any,
    candidate_count: int,
    action_dim: int,
    eval_episodes: int,
    episode_steps: int,
    seed: int,
    device: str,
    dt: float,
    frame_skip: int,
    use_cuda_graph: bool,
    wp: Any,
    mjw: Any,
) -> list[list[float]]:
    """Verify all candidates' policies in one MJWarp simulation by laying out
    (candidate_count * eval_episodes) worlds and indexing the batched policy
    parameters per row. Returns per-candidate lists of episode returns."""
    import torch

    torch_device = torch.device("cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    qpos_initial, qvel_initial = ant_reset_states(task, eval_episodes, seed)
    qpos_template = torch.as_tensor(qpos_initial, dtype=torch.float32, device=torch_device)
    qvel_template = torch.as_tensor(qvel_initial, dtype=torch.float32, device=torch_device)
    # Replicate the same initial states across candidates so verified return is
    # apples-to-apples between reward proposals.
    qpos_init_all = qpos_template.unsqueeze(0).expand(candidate_count, -1, -1).reshape(
        candidate_count * eval_episodes, -1
    )
    qvel_init_all = qvel_template.unsqueeze(0).expand(candidate_count, -1, -1).reshape(
        candidate_count * eval_episodes, -1
    )

    with torch_warp_stream(wp, torch, torch_device):
        mjw.reset_data(model, data)
        qpos = wp.to_torch(data.qpos)
        qvel = wp.to_torch(data.qvel)
        ctrl = wp.to_torch(data.ctrl)
        physics_graph = create_control_step_graph(
            model=model,
            data=data,
            mjw=mjw,
            wp=wp,
            frame_skip=frame_skip,
            enabled=use_cuda_graph,
            device=device,
        )
        qpos.copy_(qpos_init_all)
        qvel.copy_(qvel_init_all)
        qpos_view = qpos.view(candidate_count, eval_episodes, -1)
        qvel_view = qvel.view(candidate_count, eval_episodes, -1)
        ctrl_view = ctrl.view(candidate_count, eval_episodes, action_dim)
        returns = torch.zeros((candidate_count, eval_episodes), dtype=torch.float32, device=torch_device)
        dones = torch.zeros((candidate_count, eval_episodes), dtype=torch.bool, device=torch_device)
        for _step in range(episode_steps):
            obs = ant_policy_obs_torch_batched(qpos_view, qvel_view)
            qpos_xy_before = qpos_view[:, :, :2].clone()
            with torch.no_grad():
                action = policy.mean_action_all_candidates(obs)
            ctrl_view.copy_(action)
            advance_control_step(model=model, data=data, mjw=mjw, wp=wp, frame_skip=frame_skip, graph=physics_graph)
            true_reward, next_dones = original_ant_reward_torch_batched(
                qpos_xy_before, qpos_view, qvel_view, action, dt
            )
            returns.add_(torch.where(~dones, true_reward, torch.zeros_like(true_reward)))
            dones.logical_or_(next_dones)
        return returns.detach().cpu().tolist()


def original_ant_reward_torch_batched(
    qpos_xy_before: Any, qpos_after: Any, qvel_after: Any, action: Any, dt: float
) -> tuple[Any, Any]:
    import torch

    x_velocity = (qpos_after[:, :, 0] - qpos_xy_before[:, :, 0]) / dt
    torso_z = qpos_after[:, :, 2]
    healthy = torch.isfinite(qpos_after).all(dim=2) & torch.isfinite(qvel_after).all(dim=2)
    healthy &= (torso_z >= 0.2) & (torso_z <= 1.0)
    control_cost = 0.5 * 1.0e-2 * action.square().sum(dim=2)
    original_reward = x_velocity + healthy.to(dtype=torch.float32) - control_cost
    return original_reward.to(dtype=torch.float32), ~healthy


def ant_reset_states(task: str, episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env = gym.make(task)
    qpos = []
    qvel = []
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            qpos.append(np.asarray(env.unwrapped.data.qpos, dtype=np.float32).copy())
            qvel.append(np.asarray(env.unwrapped.data.qvel, dtype=np.float32).copy())
    finally:
        env.close()
    return np.asarray(qpos, dtype=np.float32), np.asarray(qvel, dtype=np.float32)


def original_ant_reward_torch(qpos_xy_before: Any, qpos_after: Any, qvel_after: Any, action: Any, dt: float) -> tuple[Any, Any]:
    import torch

    x_velocity = (qpos_after[:, 0] - qpos_xy_before[:, 0]) / dt
    torso_z = qpos_after[:, 2]
    healthy = torch.isfinite(qpos_after).all(dim=1) & torch.isfinite(qvel_after).all(dim=1)
    healthy &= (torso_z >= 0.2) & (torso_z <= 1.0)
    control_cost = 0.5 * 1.0e-2 * action.square().sum(dim=1)
    original_reward = x_velocity + healthy.to(dtype=torch.float32) - control_cost
    return original_reward.to(dtype=torch.float32), ~healthy


def evaluate_policy_in_gym(
    *,
    task: str,
    policy: tuple[str, Any],
    obs_dim: int,
    action_dim: int,
    eval_episodes: int,
    seed: int,
) -> list[float]:
    policy_type, policy_data = policy
    if policy_type == "linear":
        weights = policy_data[: obs_dim * action_dim].reshape(obs_dim, action_dim)
        biases = policy_data[obs_dim * action_dim :]
    elif policy_type == "batched_ppo":
        import torch

        torch_policy, candidate_index = policy_data
        torch_device = next(torch_policy.parameters()).device
    else:
        import torch

        torch_policy = policy_data
        torch_device = next(torch_policy.parameters()).device
    returns = []
    for episode in range(eval_episodes):
        env = gym.make(task)
        try:
            obs, _info = env.reset(seed=seed + episode)
            total_reward = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                policy_obs = np.asarray(obs[:obs_dim], dtype=np.float32)
                if policy_type == "linear":
                    action = np.tanh(policy_obs @ weights + biases).astype(np.float32)
                else:
                    obs_t = torch.as_tensor(policy_obs[None, :], dtype=torch.float32, device=torch_device)
                    with torch.no_grad():
                        if policy_type == "batched_ppo":
                            action_t = torch_policy.mean_action_for_candidate(obs_t, candidate_index)
                        else:
                            action_t = torch_policy.mean_action(obs_t)
                        action = action_t.cpu().numpy()[0].astype(np.float32)
                obs, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
        finally:
            env.close()
        returns.append(total_reward)
    return returns


class AntActorCritic:
    def __new__(cls, obs_dim: int, action_dim: int):
        import torch

        class _AntActorCritic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers = []
                last_dim = obs_dim
                for hidden_dim in (256, 128, 64):
                    layers.append(torch.nn.Linear(last_dim, hidden_dim))
                    layers.append(torch.nn.ELU())
                    last_dim = hidden_dim
                self.backbone = torch.nn.Sequential(*layers)
                self.actor_mean = torch.nn.Linear(last_dim, action_dim)
                self.critic = torch.nn.Linear(last_dim, 1)
                self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

            def forward(self, obs: Any) -> tuple[Any, Any]:
                features = self.backbone(obs)
                return self.actor_mean(features), self.critic(features).squeeze(-1)

            def distribution(self, obs: Any) -> Any:
                mean, _value = self.forward(obs)
                std = self.log_std.exp().expand_as(mean)
                return torch.distributions.Normal(mean, std)

            def act(self, obs: Any) -> tuple[Any, Any, Any, Any]:
                dist = self.distribution(obs)
                raw_action = dist.rsample()
                action = torch.tanh(raw_action)
                logprob = self.squashed_log_prob(dist.log_prob(raw_action), action)
                entropy = dist.entropy().sum(dim=-1)
                value = self.value(obs)
                return action, logprob, entropy, value

            def evaluate_actions(self, obs: Any, action: Any) -> tuple[Any, Any, Any]:
                clipped = action.clamp(-0.999, 0.999)
                raw_action = torch.atanh(clipped)
                dist = self.distribution(obs)
                logprob = self.squashed_log_prob(dist.log_prob(raw_action), clipped)
                entropy = dist.entropy().sum(dim=-1)
                value = self.value(obs)
                return logprob, entropy, value

            @staticmethod
            def squashed_log_prob(raw_log_prob: Any, action: Any) -> Any:
                correction = torch.log1p(-action.square() + 1.0e-6)
                return (raw_log_prob - correction).sum(dim=-1)

            def value(self, obs: Any) -> Any:
                _mean, value = self.forward(obs)
                return value

            def mean_action(self, obs: Any) -> Any:
                mean, _value = self.forward(obs)
                return torch.tanh(mean)

        return _AntActorCritic()


class BatchedAntActorCritic:
    def __new__(cls, candidate_count: int, obs_dim: int, action_dim: int):
        import torch
        import torch.nn.functional as functional

        class _BatchedAntActorCritic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                base = AntActorCritic(obs_dim, action_dim)
                linear_layers = [base.backbone[0], base.backbone[2], base.backbone[4], base.actor_mean, base.critic]
                for index, layer in enumerate(linear_layers):
                    setattr(
                        self,
                        f"weight_{index}",
                        torch.nn.Parameter(layer.weight.detach().unsqueeze(0).repeat(candidate_count, 1, 1)),
                    )
                    setattr(
                        self,
                        f"bias_{index}",
                        torch.nn.Parameter(layer.bias.detach().unsqueeze(0).repeat(candidate_count, 1)),
                    )
                self.log_std = torch.nn.Parameter(base.log_std.detach().unsqueeze(0).repeat(candidate_count, 1))

            def linear(self, values: Any, index: int) -> Any:
                weights = getattr(self, f"weight_{index}")
                biases = getattr(self, f"bias_{index}")
                return torch.einsum("cni,coi->cno", values, weights) + biases[:, None, :]

            def forward(self, obs: Any) -> tuple[Any, Any]:
                features = functional.elu(self.linear(obs, 0))
                features = functional.elu(self.linear(features, 1))
                features = functional.elu(self.linear(features, 2))
                return self.linear(features, 3), self.linear(features, 4).squeeze(-1)

            def act(self, obs: Any, noise: Any) -> tuple[Any, Any, Any, Any]:
                mean, value = self.forward(obs)
                std = self.log_std.exp()[:, None, :]
                raw_action = mean + std * noise
                action = torch.tanh(raw_action)
                raw_logprob = (
                    -0.5 * ((raw_action - mean) / std).square()
                    - self.log_std[:, None, :]
                    - 0.5 * math.log(2.0 * math.pi)
                )
                logprob = self.squashed_log_prob(raw_logprob, action)
                entropy = (
                    self.log_std[:, None, :] + 0.5 * (1.0 + math.log(2.0 * math.pi))
                ).expand_as(raw_action).sum(dim=-1)
                return action, logprob, entropy, value

            def evaluate_actions(self, obs: Any, action: Any) -> tuple[Any, Any, Any]:
                mean, value = self.forward(obs)
                std = self.log_std.exp()[:, None, :]
                clipped = action.clamp(-0.999, 0.999)
                raw_action = torch.atanh(clipped)
                raw_logprob = (
                    -0.5 * ((raw_action - mean) / std).square()
                    - self.log_std[:, None, :]
                    - 0.5 * math.log(2.0 * math.pi)
                )
                logprob = self.squashed_log_prob(raw_logprob, clipped)
                entropy = (
                    self.log_std[:, None, :] + 0.5 * (1.0 + math.log(2.0 * math.pi))
                ).expand_as(raw_action).sum(dim=-1)
                return logprob, entropy, value

            @staticmethod
            def squashed_log_prob(raw_log_prob: Any, action: Any) -> Any:
                correction = torch.log1p(-action.square() + 1.0e-6)
                return (raw_log_prob - correction).sum(dim=-1)

            def value(self, obs: Any) -> Any:
                _mean, value = self.forward(obs)
                return value

            def mean_action_for_candidate(self, obs: Any, candidate_index: int) -> Any:
                repeated_obs = obs.unsqueeze(0).expand(candidate_count, -1, -1)
                mean, _value = self.forward(repeated_obs)
                return torch.tanh(mean[candidate_index])

            def mean_action_all_candidates(self, obs: Any) -> Any:
                mean, _value = self.forward(obs)
                return torch.tanh(mean)

        return _BatchedAntActorCritic()


class VectorizedRewardExpression:
    def __init__(self, expression: str, allowed_names: set[str] | frozenset[str]) -> None:
        RewardExpression(expression, allowed_names)
        parsed = ast.parse(expression, mode="eval")
        parsed = NumpyWhereTransformer().visit(parsed)
        ast.fix_missing_locations(parsed)
        self._code = compile(parsed, "<vectorized-reward-expression>", "eval")
        self._scalar = RewardExpression(expression, allowed_names)

    def __call__(self, values: dict[str, np.ndarray]) -> np.ndarray:
        try:
            result = eval(self._code, {"__builtins__": {}, **NUMPY_FUNCS}, values)
            return np.clip(np.where(np.isfinite(result), result, -100.0), -100.0, 100.0).astype(np.float32)
        except Exception:
            worlds = len(next(iter(values.values())))
            rewards = np.empty(worlds, dtype=np.float32)
            for index in range(worlds):
                row = {key: value[index].item() for key, value in values.items()}
                rewards[index] = self._scalar(row)
            return rewards


class VectorizedRewardProgram:
    def __init__(
        self,
        *,
        component_expressions: dict[str, str] | None,
        expression: str,
        allowed_names: set[str] | frozenset[str],
    ) -> None:
        validate_component_expressions(component_expressions, allowed_names)
        if component_expressions is not None:
            self.component_expressions = dict(component_expressions)
        else:
            self.component_expressions = {"total": expression}
        self._components = {
            name: VectorizedRewardExpression(component_expression, allowed_names)
            for name, component_expression in self.component_expressions.items()
        }
        self.expression = total_expression_from_components(self.component_expressions, expression)

    @property
    def component_names(self) -> list[str]:
        return list(self._components)

    def components(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: expression(values) for name, expression in self._components.items()}

    def __call__(self, values: dict[str, np.ndarray]) -> np.ndarray:
        return self.total_from_components(self.components(values))

    @staticmethod
    def total_from_components(component_values: dict[str, np.ndarray]) -> np.ndarray:
        total = None
        for value in component_values.values():
            total = value.astype(np.float32) if total is None else total + value.astype(np.float32)
        if total is None:
            raise ValueError("Reward program has no components")
        return np.clip(np.where(np.isfinite(total), total, -100.0), -100.0, 100.0).astype(np.float32)


class RewardComponentTracker:
    def __init__(self, component_names: list[str]) -> None:
        self._stats = {
            name: {"sum": 0.0, "count": 0, "min": float("inf"), "max": float("-inf")}
            for name in component_names
        }

    def update(self, component_values: dict[str, np.ndarray]) -> None:
        for name, values in component_values.items():
            finite_values = np.asarray(values, dtype=np.float32)
            finite_values = finite_values[np.isfinite(finite_values)]
            if finite_values.size == 0:
                continue
            stats = self._stats.setdefault(
                name, {"sum": 0.0, "count": 0, "min": float("inf"), "max": float("-inf")}
            )
            stats["sum"] += float(np.sum(finite_values))
            stats["count"] += int(finite_values.size)
            stats["min"] = min(float(stats["min"]), float(np.min(finite_values)))
            stats["max"] = max(float(stats["max"]), float(np.max(finite_values)))

    def summary(self) -> dict[str, dict[str, float]]:
        summary = {}
        for name, stats in self._stats.items():
            count = int(stats["count"])
            if count == 0:
                continue
            summary[name] = {
                "mean": float(stats["sum"]) / count,
                "min": float(stats["min"]),
                "max": float(stats["max"]),
            }
        return summary


class TorchRewardExpression:
    def __init__(self, expression: str, allowed_names: set[str] | frozenset[str]) -> None:
        RewardExpression(expression, allowed_names)
        parsed = ast.parse(expression, mode="eval")
        parsed = TorchWhereTransformer().visit(parsed)
        ast.fix_missing_locations(parsed)
        self._code = compile(parsed, "<torch-reward-expression>", "eval")

    def __call__(self, values: dict[str, Any]) -> Any:
        import torch

        result = eval(self._code, {"__builtins__": {}, **torch_functions()}, {**values, "pi": math.pi})
        template = next(iter(values.values()))
        if not torch.is_tensor(result):
            result = torch.full_like(template, float(result), dtype=torch.float32)
        result = result.to(dtype=torch.float32)
        return torch.clamp(
            torch.where(torch.isfinite(result), result, torch.full_like(result, -100.0)),
            -100.0,
            100.0,
        )


class TorchRewardProgram:
    def __init__(
        self,
        *,
        component_expressions: dict[str, str] | None,
        expression: str,
        allowed_names: set[str] | frozenset[str],
    ) -> None:
        validate_component_expressions(component_expressions, allowed_names)
        self.component_expressions = dict(component_expressions) if component_expressions is not None else {"total": expression}
        self._components = {
            name: TorchRewardExpression(component_expression, allowed_names)
            for name, component_expression in self.component_expressions.items()
        }
        self.expression = total_expression_from_components(self.component_expressions, expression)

    @property
    def component_names(self) -> list[str]:
        return list(self._components)

    def components(self, values: dict[str, Any]) -> dict[str, Any]:
        return {name: expression(values) for name, expression in self._components.items()}

    @staticmethod
    def total_from_components(component_values: dict[str, Any]) -> Any:
        import torch

        values = list(component_values.values())
        if not values:
            raise ValueError("Reward program has no components")
        total = torch.stack([value.to(dtype=torch.float32) for value in values]).sum(dim=0)
        return torch.clamp(
            torch.where(torch.isfinite(total), total, torch.full_like(total, -100.0)),
            -100.0,
            100.0,
        )


class CompiledTorchRewardProgram:
    """Optional compiled wrapper retaining the eager reward program as fallback."""

    def __init__(self, eager: TorchRewardProgram) -> None:
        import torch

        self._eager = eager
        self.component_expressions = eager.component_expressions
        self.expression = eager.expression
        self._compiled = torch.compile(eager.components, dynamic=True)
        self._disabled = False

    @property
    def component_names(self) -> list[str]:
        return self._eager.component_names

    def components(self, values: dict[str, Any]) -> dict[str, Any]:
        if self._disabled:
            return self._eager.components(values)
        try:
            return self._compiled(values)
        except Exception:
            self._disabled = True
            return self._eager.components(values)

    @staticmethod
    def total_from_components(component_values: dict[str, Any]) -> Any:
        return TorchRewardProgram.total_from_components(component_values)


def make_torch_reward_program(
    *,
    component_expressions: dict[str, str] | None,
    expression: str,
    allowed_names: set[str] | frozenset[str],
    backend: str,
) -> TorchRewardProgram | CompiledTorchRewardProgram:
    validate_reward_backend(backend)
    eager = TorchRewardProgram(
        component_expressions=component_expressions,
        expression=expression,
        allowed_names=allowed_names,
    )
    return eager if backend == "eager" else CompiledTorchRewardProgram(eager)


class TorchRewardComponentTracker:
    def __init__(self, component_names: list[str], device: Any) -> None:
        import torch

        self._stats = {
            name: {
                "sum": torch.zeros((), dtype=torch.float32, device=device),
                "count": torch.zeros((), dtype=torch.int64, device=device),
                "min": torch.full((), float("inf"), dtype=torch.float32, device=device),
                "max": torch.full((), float("-inf"), dtype=torch.float32, device=device),
            }
            for name in component_names
        }

    def update(self, component_values: dict[str, Any]) -> None:
        import torch

        for name, values in component_values.items():
            finite = torch.isfinite(values)
            stats = self._stats[name]
            stats["sum"].add_(torch.where(finite, values, torch.zeros_like(values)).sum())
            stats["count"].add_(finite.sum())
            stats["min"] = torch.minimum(
                stats["min"], torch.where(finite, values, torch.full_like(values, float("inf"))).min()
            )
            stats["max"] = torch.maximum(
                stats["max"], torch.where(finite, values, torch.full_like(values, float("-inf"))).max()
            )

    def summary(self) -> dict[str, dict[str, float]]:
        return _component_snapshot_to_summary(self.snapshot())

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            name: {key: value.detach().clone() for key, value in stats.items()}
            for name, stats in self._stats.items()
        }


def _component_snapshot_to_summary(snapshot: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, stats in snapshot.items():
        count = int(stats["count"].item()) if hasattr(stats["count"], "item") else int(stats["count"])
        if not count:
            continue
        sum_val = float(stats["sum"].item()) if hasattr(stats["sum"], "item") else float(stats["sum"])
        min_val = float(stats["min"].item()) if hasattr(stats["min"], "item") else float(stats["min"])
        max_val = float(stats["max"].item()) if hasattr(stats["max"], "item") else float(stats["max"])
        summary[name] = {"mean": sum_val / count, "min": min_val, "max": max_val}
    return summary


def _materialize_metric_log(pending: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not pending:
        return []
    import torch

    mean_shaped = torch.stack([row["mean_shaped_t"] for row in pending]).detach().cpu().tolist()
    max_shaped = torch.stack([row["max_shaped_t"] for row in pending]).detach().cpu().tolist()
    mean_true = torch.stack([row["mean_true_t"] for row in pending]).detach().cpu().tolist()
    best_true = torch.stack([row["best_true_t"] for row in pending]).detach().cpu().tolist()
    summaries = []
    for index, row in enumerate(pending):
        summaries.append(
            {
                "iteration": row["iteration"],
                "ppo_update": row["ppo_update"],
                "mean_shaped_return": float(mean_shaped[index]),
                "best_shaped_return": float(max_shaped[index]),
                "mean_true_return_in_population": float(mean_true[index]),
                "best_true_return_in_population": float(best_true[index]),
                "reward_component_stats": _component_snapshot_to_summary(row["component_snapshot"]),
            }
        )
    return summaries


def _materialize_metric_log_batched(
    pending: list[dict[str, Any]], candidate_count: int
) -> list[list[dict[str, Any]]]:
    summaries: list[list[dict[str, Any]]] = [[] for _ in range(candidate_count)]
    if not pending:
        return summaries
    import torch

    mean_shaped = torch.stack([row["mean_shaped_t"] for row in pending]).detach().cpu().tolist()
    max_shaped = torch.stack([row["max_shaped_t"] for row in pending]).detach().cpu().tolist()
    mean_true = torch.stack([row["mean_true_t"] for row in pending]).detach().cpu().tolist()
    best_true = torch.stack([row["best_true_t"] for row in pending]).detach().cpu().tolist()
    for chunk_index, row in enumerate(pending):
        for candidate_index in range(candidate_count):
            summaries[candidate_index].append(
                {
                    "iteration": row["iteration"],
                    "ppo_update": row["ppo_update"],
                    "mean_shaped_return": float(mean_shaped[chunk_index][candidate_index]),
                    "best_shaped_return": float(max_shaped[chunk_index][candidate_index]),
                    "mean_true_return_in_population": float(mean_true[chunk_index][candidate_index]),
                    "best_true_return_in_population": float(best_true[chunk_index][candidate_index]),
                    "reward_component_stats": _component_snapshot_to_summary(
                        row["component_snapshots"][candidate_index]
                    ),
                }
            )
    return summaries


class NumpyWhereTransformer(ast.NodeTransformer):
    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        return ast.Call(
            func=ast.Name(id="where", ctx=ast.Load()),
            args=[self.visit(node.test), self.visit(node.body), self.visit(node.orelse)],
            keywords=[],
        )


class TorchWhereTransformer(ast.NodeTransformer):
    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        return ast.Call(
            func=ast.Name(id="where", ctx=ast.Load()),
            args=[self.visit(node.test), self.visit(node.body), self.visit(node.orelse)],
            keywords=[],
        )


NUMPY_FUNCS = {
    **ALLOWED_FUNCS,
    "abs": np.abs,
    "min": np.minimum,
    "max": np.maximum,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
    "exp": np.exp,
    "where": np.where,
}


def torch_functions() -> dict[str, Any]:
    import torch

    def as_compatible_tensor(value: Any, like: Any) -> Any:
        return value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32, device=like.device)

    def unary(op: Any, scalar_op: Any, value: Any) -> Any:
        return op(value) if torch.is_tensor(value) else scalar_op(value)

    def pairwise(op: Any, scalar_op: Any, left: Any, right: Any) -> Any:
        if not torch.is_tensor(left) and not torch.is_tensor(right):
            return scalar_op(left, right)
        template = left if torch.is_tensor(left) else right
        return op(as_compatible_tensor(left, template), as_compatible_tensor(right, template))

    return {
        "abs": lambda value: unary(torch.abs, builtins.abs, value),
        "min": lambda left, right: pairwise(torch.minimum, builtins.min, left, right),
        "max": lambda left, right: pairwise(torch.maximum, builtins.max, left, right),
        "float": lambda value: value.to(dtype=torch.float32) if torch.is_tensor(value) else float(value),
        "sqrt": lambda value: unary(torch.sqrt, math.sqrt, value),
        "sin": lambda value: unary(torch.sin, math.sin, value),
        "cos": lambda value: unary(torch.cos, math.cos, value),
        "tanh": lambda value: unary(torch.tanh, math.tanh, value),
        "exp": lambda value: unary(torch.exp, math.exp, value),
        "where": torch.where,
    }
