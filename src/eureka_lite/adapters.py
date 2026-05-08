from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class TaskAdapter(Protocol):
    task_id: str
    prompt_id: str
    reward_variables: frozenset[str]

    def reward_context(
        self,
        obs: np.ndarray,
        action: Any,
        original_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, float | bool]:
        ...


def scalar_action_l2(action: Any) -> float:
    return float(np.square(np.asarray(action, dtype=np.float32)).sum())


@dataclass(frozen=True)
class CartPoleAdapter:
    task_id: str = "CartPole-v1"
    prompt_id: str = "cartpole_reward_design_v1"
    reward_variables: frozenset[str] = frozenset(
        {
            "x",
            "x_dot",
            "theta",
            "theta_dot",
            "action",
            "original_reward",
            "terminated",
            "truncated",
        }
    )

    def reward_context(
        self,
        obs: np.ndarray,
        action: Any,
        original_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, float | bool]:
        del info
        x, x_dot, theta, theta_dot = [float(v) for v in obs]
        action_value = float(np.asarray(action).reshape(-1)[0])
        return {
            "x": x,
            "x_dot": x_dot,
            "theta": theta,
            "theta_dot": theta_dot,
            "action": action_value,
            "original_reward": float(original_reward),
            "terminated": terminated,
            "truncated": truncated,
        }


@dataclass(frozen=True)
class AntAdapter:
    task_id: str = "Ant-v5"
    prompt_id: str = "ant_reward_design_v1"
    reward_variables: frozenset[str] = frozenset(
        {
            "x_velocity",
            "y_velocity",
            "forward_reward",
            "control_cost",
            "survive_reward",
            "torso_z",
            "action_l2",
            "original_reward",
            "healthy",
            "terminated",
            "truncated",
        }
    )

    def reward_context(
        self,
        obs: np.ndarray,
        action: Any,
        original_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, float | bool]:
        # In Gymnasium Ant, observation[0] is torso z when xy position is excluded.
        torso_z = float(np.asarray(obs, dtype=np.float32).reshape(-1)[0])
        forward_reward = float(info.get("forward_reward", info.get("reward_forward", 0.0)))
        control_cost = abs(float(info.get("reward_ctrl", 0.0)))
        survive_reward = float(info.get("reward_survive", 0.0))
        return {
            "x_velocity": float(info.get("x_velocity", forward_reward)),
            "y_velocity": float(info.get("y_velocity", 0.0)),
            "forward_reward": forward_reward,
            "control_cost": control_cost,
            "survive_reward": survive_reward,
            "torso_z": torso_z,
            "action_l2": scalar_action_l2(action),
            "original_reward": float(original_reward),
            "healthy": not terminated,
            "terminated": terminated,
            "truncated": truncated,
        }


ADAPTERS: dict[str, TaskAdapter] = {
    "CartPole-v1": CartPoleAdapter(),
    "Ant-v5": AntAdapter(),
}


def get_adapter(task: str) -> TaskAdapter:
    try:
        return ADAPTERS[task]
    except KeyError as exc:
        supported = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unsupported task {task!r}. Supported tasks: {supported}") from exc

