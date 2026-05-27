from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from eureka_lite.adapters import AntAdapter
from eureka_lite.mjwarp_evaluator import (
    BatchedAntActorCritic,
    MjwarpEvaluatorConfig,
    AntActorCritic,
    TorchRewardProgram,
    VectorizedRewardExpression,
    VectorizedRewardProgram,
    _materialize_metric_log,
    _materialize_metric_log_batched,
    advance_control_step,
    compute_gae,
    compute_gae_torch,
    build_verification_audit,
    make_torch_reward_program,
    mjwarp_control_step,
    reset_mjwarp_worlds,
    seed_torch_policy_rng,
    load_base_policy_into_batched,
    load_base_policy_into_single,
    train_and_evaluate_mjwarp,
    validate_ppo_init_config,
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

    def test_compiled_reward_backend_preserves_eager_semantics(self) -> None:
        import torch

        values = {
            "x_velocity": torch.tensor([1.0, 2.0]),
            "survive_reward": torch.tensor([1.0, 1.0]),
        }
        with patch("torch.compile", side_effect=lambda function, **_kwargs: function):
            program = make_torch_reward_program(
                component_expressions={"score": "x_velocity + survive_reward"},
                expression="",
                allowed_names=AntAdapter().reward_variables,
                backend="compiled",
            )
        total = program.total_from_components(program.components(values))
        self.assertTrue(torch.equal(total, torch.tensor([2.0, 3.0])))

    def test_verification_audit_records_difference_and_enforces_threshold(self) -> None:
        config = MjwarpEvaluatorConfig(verified_audit_gym=True, verified_audit_max_abs_diff=0.2)
        audit = build_verification_audit([1.0, 2.0], [1.1, 1.9], config)
        self.assertTrue(audit["passed"])
        self.assertAlmostEqual(audit["max_abs_diff"], 0.1)
        with self.assertRaisesRegex(RuntimeError, "audit failed"):
            build_verification_audit([1.0], [2.0], config)

    def test_gpu_metric_materialization_exposes_best_internal_true_return(self) -> None:
        import torch

        row = {
            "iteration": 0,
            "ppo_update": 0,
            "mean_shaped_t": torch.tensor(1.0),
            "max_shaped_t": torch.tensor(2.0),
            "mean_true_t": torch.tensor(3.0),
            "best_true_t": torch.tensor(4.0),
            "component_snapshot": {},
        }
        batched_row = {
            **row,
            "mean_shaped_t": torch.tensor([1.0, 1.5]),
            "max_shaped_t": torch.tensor([2.0, 2.5]),
            "mean_true_t": torch.tensor([3.0, 3.5]),
            "best_true_t": torch.tensor([4.0, 4.5]),
            "component_snapshots": [{}, {}],
        }
        self.assertEqual(_materialize_metric_log([row])[0]["best_true_return_in_population"], 4.0)
        self.assertEqual(_materialize_metric_log_batched([batched_row], 2)[1][0]["best_true_return_in_population"], 4.5)

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

    def test_reset_mjwarp_worlds_passes_flat_device_mask(self) -> None:
        import torch

        class FakeWarp:
            bool = bool

            def __init__(self) -> None:
                self.mask = None

            def from_torch(self, mask, dtype=None):
                self.mask = (mask, dtype)
                return "warp-mask"

        class FakeMjwarp:
            def __init__(self) -> None:
                self.reset = None

            def reset_data(self, model, data, reset=None) -> None:
                self.reset = (model, data, reset)

        wp = FakeWarp()
        mjw = FakeMjwarp()
        reset_mjwarp_worlds("m", "d", torch.tensor([[True, False], [False, True]]), mjw=mjw, wp=wp)
        self.assertEqual(wp.mask[0].tolist(), [True, False, False, True])
        self.assertEqual(mjw.reset, ("m", "d", "warp-mask"))

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

    def test_base_policy_checkpoint_loads_single_policy(self) -> None:
        import torch

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "base.pt"
            source = AntActorCritic(4, 2)
            with torch.no_grad():
                source.log_std.fill_(0.25)
            torch.save(
                {
                    "policy_state_dict": source.state_dict(),
                    "metadata": {"obs_dim": 4, "action_dim": 2},
                },
                path,
            )
            target = AntActorCritic(4, 2)
            load_base_policy_into_single(target, path, obs_dim=4, action_dim=2, torch_device=torch.device("cpu"))
            for key, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[key]))

    def test_base_policy_checkpoint_replicates_into_batched_policy(self) -> None:
        import torch

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "base.pt"
            source = AntActorCritic(4, 2)
            with torch.no_grad():
                source.actor_mean.bias.fill_(0.42)
                source.log_std.fill_(0.25)
            torch.save(
                {
                    "policy_state_dict": source.state_dict(),
                    "metadata": {"obs_dim": 4, "action_dim": 2},
                },
                path,
            )
            target = BatchedAntActorCritic(3, 4, 2)
            load_base_policy_into_batched(
                target,
                path,
                candidate_count=3,
                obs_dim=4,
                action_dim=2,
                torch_device=torch.device("cpu"),
            )
            self.assertTrue(torch.equal(target.bias_3[0], source.actor_mean.bias))
            self.assertTrue(torch.equal(target.bias_3[0], target.bias_3[1]))
            self.assertTrue(torch.equal(target.bias_3[1], target.bias_3[2]))
            self.assertTrue(torch.equal(target.log_std[0], source.log_std))
            self.assertTrue(torch.equal(target.log_std[0], target.log_std[2]))

    def test_base_init_requires_checkpoint_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "base_policy_checkpoint"):
            validate_ppo_init_config(MjwarpEvaluatorConfig(ppo_init_mode="base"))

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
