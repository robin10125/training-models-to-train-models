from __future__ import annotations

import unittest

import numpy as np

from eureka_lite.adapters import AntAdapter
from eureka_lite.rewards import RewardExpression


class RewardExpressionTests(unittest.TestCase):
    def test_ant_expression_evaluates_from_adapter_variables(self) -> None:
        adapter = AntAdapter()
        expression = RewardExpression(
            "1.0 * x_velocity + survive_reward - 0.1 * action_l2 - (2.0 if terminated else 0.0)",
            adapter.reward_variables,
        )
        reward = expression(
            {
                "x_velocity": 2.0,
                "survive_reward": 1.0,
                "action_l2": 3.0,
                "terminated": False,
            }
        )
        self.assertAlmostEqual(reward, 2.7)

    def test_expression_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            RewardExpression("__import__('os').system('true')", AntAdapter().reward_variables)


class AntAdapterTests(unittest.TestCase):
    def test_ant_adapter_extracts_reward_context(self) -> None:
        adapter = AntAdapter()
        context = adapter.reward_context(
            obs=np.array([0.52, 1.0, 0.0], dtype=np.float32),
            action=np.array([1.0, -2.0], dtype=np.float32),
            original_reward=1.25,
            terminated=False,
            truncated=False,
            info={
                "x_velocity": 1.5,
                "y_velocity": -0.25,
                "forward_reward": 1.5,
                "reward_ctrl": -0.5,
                "reward_survive": 1.0,
            },
        )
        self.assertEqual(context["x_velocity"], 1.5)
        self.assertEqual(context["control_cost"], 0.5)
        self.assertEqual(context["action_l2"], 5.0)
        self.assertEqual(context["original_reward"], 1.25)
        self.assertTrue(context["healthy"])


if __name__ == "__main__":
    unittest.main()

