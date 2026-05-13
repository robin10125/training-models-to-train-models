from __future__ import annotations

import unittest

import numpy as np

from eureka_lite.adapters import AntAdapter
from eureka_lite.mjwarp_evaluator import MjwarpEvaluatorConfig, VectorizedRewardExpression, train_and_evaluate_mjwarp
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

    def test_rejects_non_ant_task_before_importing_mjwarp(self) -> None:
        candidate = RewardCandidate(
            name="c",
            task="CartPole-v1",
            prompt_id="p",
            prompt="p",
            expression="original_reward",
            weights={},
        )
        with self.assertRaisesRegex(ValueError, "Ant-v5"):
            train_and_evaluate_mjwarp(candidate, MjwarpEvaluatorConfig(task="CartPole-v1"))


if __name__ == "__main__":
    unittest.main()
