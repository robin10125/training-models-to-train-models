from __future__ import annotations

import unittest

import numpy as np

from eureka_lite.adapters import AntAdapter
from eureka_lite.mjwarp_evaluator import (
    BatchedAntActorCritic,
    MjwarpEvaluatorConfig,
    TorchRewardProgram,
    VectorizedRewardExpression,
    VectorizedRewardProgram,
    advance_control_step,
    compute_gae,
    compute_gae_torch,
    mjwarp_control_step,
    seed_torch_policy_rng,
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

    def test_torch_reward_program_matches_numpy_program(self) -> None:
        import torch

        components = {
            "forward": "x_velocity",
            "alive": "survive_reward",
            "control": "-0.1 * action_l2 - (2.0 if terminated else 0.0)",
            "constant_math": "0.0 * x_velocity + max(0.1, abs(-0.2))",
        }
        numpy_program = VectorizedRewardProgram(
            component_expressions=components,
            expression="",
            allowed_names=AntAdapter().reward_variables,
        )
        torch_program = TorchRewardProgram(
            component_expressions=components,
            expression="",
            allowed_names=AntAdapter().reward_variables,
        )
        numpy_values = {
            "x_velocity": np.array([1.0, 2.0], dtype=np.float32),
            "survive_reward": np.array([1.0, 1.0], dtype=np.float32),
            "action_l2": np.array([3.0, 4.0], dtype=np.float32),
            "terminated": np.array([False, True]),
        }
        torch_values = {key: torch.as_tensor(value) for key, value in numpy_values.items()}
        numpy_total = numpy_program.total_from_components(numpy_program.components(numpy_values))
        torch_total = torch_program.total_from_components(torch_program.components(torch_values)).numpy()
        self.assertTrue(np.allclose(torch_total, numpy_total))

    def test_torch_gae_matches_numpy_gae(self) -> None:
        import torch

        rewards = np.array([[1.0, 2.0], [3.0, 1.0]], dtype=np.float32)
        values = np.array([[0.2, 0.5], [0.4, 0.6]], dtype=np.float32)
        dones = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        next_value = np.array([0.0, 0.7], dtype=np.float32)
        expected = compute_gae(
            rewards=rewards, values=values, dones=dones, next_value=next_value, gamma=0.99, gae_lambda=0.95
        )
        actual = compute_gae_torch(
            rewards=torch.as_tensor(rewards),
            values=torch.as_tensor(values),
            dones=torch.as_tensor(dones),
            next_value=torch.as_tensor(next_value),
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertTrue(np.allclose(actual[0].numpy(), expected[0]))
        self.assertTrue(np.allclose(actual[1].numpy(), expected[1]))

    def test_seed_torch_policy_rng_repeats_stochastic_draws(self) -> None:
        import torch

        seed_torch_policy_rng(17)
        first = torch.rand(4)
        seed_torch_policy_rng(17)
        second = torch.rand(4)
        self.assertTrue(torch.equal(first, second))

    def test_control_step_applies_ant_frame_skip(self) -> None:
        class FakeMjwarp:
            def __init__(self) -> None:
                self.steps = 0

            def step(self, _model, _data) -> None:
                self.steps += 1

        mjw = FakeMjwarp()
        mjwarp_control_step(None, None, mjw=mjw, frame_skip=5)
        self.assertEqual(mjw.steps, 5)

    def test_advance_control_step_uses_captured_graph(self) -> None:
        class FakeWarp:
            def __init__(self) -> None:
                self.graphs = []

            def capture_launch(self, graph) -> None:
                self.graphs.append(graph)

        class FakeMjwarp:
            def step(self, _model, _data) -> None:
                raise AssertionError("direct stepping should not be used with a graph")

        wp = FakeWarp()
        advance_control_step(model=None, data=None, mjw=FakeMjwarp(), wp=wp, frame_skip=5, graph="g")
        self.assertEqual(wp.graphs, ["g"])

    def test_batched_policy_starts_candidates_from_identical_parameters(self) -> None:
        import torch

        seed_torch_policy_rng(17)
        policy = BatchedAntActorCritic(2, 4, 2)
        obs = torch.randn(1, 3, 4).expand(2, -1, -1)
        noise = torch.randn(1, 3, 2).expand(2, -1, -1)
        with torch.no_grad():
            actions, logprobs, _entropy, values = policy.act(obs, noise)
        self.assertTrue(torch.equal(actions[0], actions[1]))
        self.assertTrue(torch.equal(logprobs[0], logprobs[1]))
        self.assertTrue(torch.equal(values[0], values[1]))

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
    advance_control_step,
