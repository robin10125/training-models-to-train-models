from __future__ import annotations

import unittest

import numpy as np

from eureka_lite.adapters import AntAdapter
from eureka_lite.mjwarp_evaluator import (
    MjwarpEvaluatorConfig,
    VectorizedRewardExpression,
    VectorizedRewardProgram,
    train_and_evaluate_mjwarp,
)
from eureka_lite.rewards import RewardCandidate


class MjwarpEvaluatorTests(unittest.TestCase):
    def test_vectorized_reward_expression_handles_ant_ternary(self) -> None:
        expression = VectorizedRewardExpression(
            "x_velocity + survive_reward - 0.1 * action_l2 - (2.0 if terminated else 0.0)",
            AntAdapter().reward_variables,
        )
        values = {
            "x_velocity": np.array([1.0, 2.0], dtype=np.float32),
            "survive_reward": np.array([1.0, 1.0], dtype=np.float32),
            "action_l2": np.array([3.0, 4.0], dtype=np.float32),
            "terminated": np.array([False, True]),
        }
        rewards = expression(values)
        self.assertTrue(np.allclose(rewards, [1.7, 0.6]))

    def test_vectorized_reward_program_returns_components_and_total(self) -> None:
        program = VectorizedRewardProgram(
            component_expressions={
                "forward": "x_velocity",
                "alive": "survive_reward",
                "control": "-0.1 * action_l2",
            },
            expression="",
            allowed_names=AntAdapter().reward_variables,
        )
        values = {
            "x_velocity": np.array([1.0, 2.0], dtype=np.float32),
            "survive_reward": np.array([1.0, 1.0], dtype=np.float32),
            "action_l2": np.array([3.0, 4.0], dtype=np.float32),
        }
        components = program.components(values)
        self.assertIn("forward", components)
        self.assertTrue(np.allclose(program.total_from_components(components), [1.7, 2.6]))

    def test_vectorized_reward_program_rejects_empty_components(self) -> None:
        with self.assertRaises(ValueError):
            VectorizedRewardProgram(
                component_expressions={},
                expression="x_velocity",
                allowed_names=AntAdapter().reward_variables,
            )

    def test_rejects_non_ant_task_before_importing_mjwarp(self) -> None:
        candidate = RewardCandidate(
            name="c",
            task="Unknown-v0",
            prompt_id="p",
            prompt="p",
            expression="original_reward",
            weights={},
        )
        with self.assertRaisesRegex(ValueError, "Ant-v5"):
            train_and_evaluate_mjwarp(candidate, MjwarpEvaluatorConfig(task="Unknown-v0"))


if __name__ == "__main__":
    unittest.main()
