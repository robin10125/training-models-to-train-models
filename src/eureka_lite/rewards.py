from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .adapters import TaskAdapter


ALLOWED_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tanh": math.tanh,
    "exp": math.exp,
}

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


@dataclass(frozen=True)
class RewardCandidate:
    name: str
    task: str
    prompt_id: str
    prompt: str
    expression: str
    weights: dict[str, float]
    component_expressions: dict[str, str] | None = None
    completion_text: str | None = None
    generation: int = 0
    generator_type: str = "mock"
    generator_checkpoint: str = "mock-v1"
    completion_token_ids: list[int] | None = None
    old_logprobs: list[float] | None = None
    eureka_role: str = "initial"
    eureka_parent_names: list[str] | None = None
    eureka_parent_expressions: list[str] | None = None
    eureka_parent_scores: list[float | None] | None = None
    eureka_elite_names: list[str] | None = None
    eureka_elite_expressions: list[str] | None = None
    eureka_elite_scores: list[float | None] | None = None
    eureka_feedback: str | None = None


class RewardExpression:
    def __init__(self, expression: str, allowed_names: set[str] | frozenset[str]) -> None:
        self.expression = expression
        self.allowed_names = set(allowed_names) | {"pi"}
        parsed = ast.parse(expression, mode="eval")
        self._validate(parsed)
        self._code = compile(parsed, "<reward-expression>", "eval")

    def __call__(self, values: dict[str, float | bool]) -> float:
        values = {**values, "pi": math.pi}
        result = eval(self._code, {"__builtins__": {}, **ALLOWED_FUNCS}, values)
        if not np.isfinite(result):
            return -100.0
        return float(np.clip(result, -100.0, 100.0))

    def _validate(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if not isinstance(child, ALLOWED_NODES):
                raise ValueError(f"Disallowed reward syntax: {type(child).__name__}")
            if isinstance(child, ast.Name) and child.id not in self.allowed_names and child.id not in ALLOWED_FUNCS:
                raise ValueError(f"Disallowed reward name: {child.id}")
            if isinstance(child, ast.Call):
                if not isinstance(child.func, ast.Name) or child.func.id not in ALLOWED_FUNCS:
                    raise ValueError("Reward expressions can only call approved math helpers")


def total_expression_from_components(component_expressions: dict[str, str] | None, fallback: str) -> str:
    if not component_expressions:
        return fallback
    return " + ".join(f"({expression})" for expression in component_expressions.values())


def validate_component_expressions(
    component_expressions: dict[str, str] | None, allowed_names: set[str] | frozenset[str]
) -> None:
    if component_expressions is None:
        return
    if not component_expressions:
        raise ValueError("Reward program must include at least one component")
    for name, expression in component_expressions.items():
        if not name.isidentifier():
            raise ValueError(f"Reward component name must be a Python identifier: {name!r}")
        RewardExpression(expression, allowed_names)


class RewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, expression: str, adapter: TaskAdapter) -> None:
        super().__init__(env)
        self.adapter = adapter
        self.reward_expression = RewardExpression(expression, adapter.reward_variables)

    def step(self, action: Any):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        values = self.adapter.reward_context(obs, action, original_reward, terminated, truncated, info)
        shaped_reward = self.reward_expression(values)
        return obs, shaped_reward, terminated, truncated, info
