from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .adapters import get_adapter
from .rewards import ALLOWED_FUNCS, RewardCandidate, RewardExpression


@dataclass(frozen=True)
class MjwarpEvaluatorConfig:
    task: str = "Ant-v5"
    worlds_per_candidate: int = 4096
    episode_steps: int = 500
    policy_iterations: int = 4
    elite_frac: float = 0.1
    init_std: float = 0.35
    min_std: float = 0.03
    seed: int = 7
    device: str = "cuda:0"
    eval_episodes: int = 5


def train_and_evaluate_mjwarp(candidate: RewardCandidate, config: MjwarpEvaluatorConfig) -> dict[str, Any]:
    if config.task != "Ant-v5":
        raise ValueError("The MJWarp evaluator currently supports only Ant-v5")
    if config.worlds_per_candidate < 2:
        raise ValueError("worlds_per_candidate must be at least 2")
    if config.episode_steps < 1:
        raise ValueError("episode_steps must be at least 1")
    if config.policy_iterations < 1:
        raise ValueError("policy_iterations must be at least 1")
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
    reward_expression = VectorizedRewardExpression(candidate.expression, adapter.reward_variables)
    rng = np.random.default_rng(config.seed)

    env = gym.make(config.task)
    try:
        mjm = env.unwrapped.model
        dt = float(mjm.opt.timestep * env.unwrapped.frame_skip)
        action_dim = int(mjm.nu)
        obs_dim = int(mjm.nq - 2 + mjm.nv)
        param_dim = obs_dim * action_dim + action_dim
        mean = np.zeros(param_dim, dtype=np.float32)
        std = np.full(param_dim, config.init_std, dtype=np.float32)
        best_params = mean.copy()
        best_shaped_return = float("-inf")
        iteration_summaries = []

        wp.init()
        with wp.ScopedDevice(config.device):
            model = mjw.put_model(mjm)
            data = mjw.make_data(mjm, nworld=config.worlds_per_candidate)

            for iteration in range(config.policy_iterations):
                params = rng.normal(mean, std, size=(config.worlds_per_candidate, param_dim)).astype(np.float32)
                shaped_returns, true_returns = rollout_policy_population(
                    model=model,
                    data=data,
                    mjm=mjm,
                    params=params,
                    reward_expression=reward_expression,
                    episode_steps=config.episode_steps,
                    dt=dt,
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
                    }
                )
    finally:
        env.close()

    eval_returns = evaluate_policy_in_gym(
        task=config.task,
        params=best_params,
        obs_dim=obs_dim,
        action_dim=action_dim,
        eval_episodes=config.eval_episodes,
        seed=config.seed + 10_000,
    )
    return {
        "mean_reward": float(np.mean(eval_returns)),
        "std_reward": float(np.std(eval_returns)),
        "episode_rewards": eval_returns,
        "elapsed_seconds": time.monotonic() - started_at,
        "metadata": {
            "sim_backend": "mjwarp",
            "worlds_per_candidate": config.worlds_per_candidate,
            "episode_steps": config.episode_steps,
            "policy_iterations": config.policy_iterations,
            "training_world_steps": config.worlds_per_candidate * config.episode_steps * config.policy_iterations,
            "elite_frac": config.elite_frac,
            "best_shaped_return": best_shaped_return,
            "iteration_summaries": iteration_summaries,
        },
    }


def rollout_policy_population(
    *,
    model: Any,
    data: Any,
    mjm: Any,
    params: np.ndarray,
    reward_expression: "VectorizedRewardExpression",
    episode_steps: int,
    dt: float,
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

    mjw.reset_data(model, data)
    wp.synchronize()
    for _step in range(episode_steps):
        qpos_before = np.asarray(data.qpos.numpy(), dtype=np.float32)
        qvel_before = np.asarray(data.qvel.numpy(), dtype=np.float32)
        obs = ant_policy_obs(qpos_before, qvel_before)
        action = np.tanh(np.einsum("wo,woa->wa", obs, weights) + biases).astype(np.float32)
        wp.copy(data.ctrl, wp.array(action, dtype=wp.float32, device=device))
        mjw.step(model, data)
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
        shaped_reward = reward_expression(context)
        shaped_returns += np.where(active, shaped_reward, 0.0).astype(np.float32)
        true_returns += np.where(active, original_reward, 0.0).astype(np.float32)
        if terminated.all():
            break
    return shaped_returns, true_returns


def ant_policy_obs(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    return np.concatenate([qpos[:, 2:], qvel], axis=1).astype(np.float32)


def evaluate_policy_in_gym(
    *,
    task: str,
    params: np.ndarray,
    obs_dim: int,
    action_dim: int,
    eval_episodes: int,
    seed: int,
) -> list[float]:
    weights = params[: obs_dim * action_dim].reshape(obs_dim, action_dim)
    biases = params[obs_dim * action_dim :]
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
                action = np.tanh(policy_obs @ weights + biases).astype(np.float32)
                obs, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
        finally:
            env.close()
        returns.append(total_reward)
    return returns


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


class NumpyWhereTransformer(ast.NodeTransformer):
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
