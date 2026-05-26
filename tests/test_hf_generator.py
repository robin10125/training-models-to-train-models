from __future__ import annotations

import unittest

from eureka_lite.hf_generator import (
    build_reward_prompt,
    extract_reward_components,
    extract_reward_expression,
    token_logprobs,
)


class HfGeneratorHelperTests(unittest.TestCase):
    def test_extract_reward_expression_from_plain_text(self) -> None:
        expression = extract_reward_expression("1.0 * x_velocity - 0.01 * action_l2")
        self.assertEqual(expression, "1.0 * x_velocity - 0.01 * action_l2")

    def test_extract_reward_expression_from_assignment(self) -> None:
        expression = extract_reward_expression("REWARD_EXPRESSION = 1.0 * x_velocity")
        self.assertEqual(expression, "1.0 * x_velocity")

    def test_extract_reward_components_from_dict_literal(self) -> None:
        components = extract_reward_components(
            "{'forward': 'x_velocity', 'control': '-0.01 * action_l2'}"
        )
        self.assertEqual(components["forward"], "x_velocity")
        self.assertEqual(components["control"], "-0.01 * action_l2")

    def test_build_reward_prompt_names_variables(self) -> None:
        prompt = build_reward_prompt(
            task="Ant-v5",
            reward_variables=["action_l2", "survive_reward", "x_velocity"],
            best_expression="x_velocity",
            best_score=10.0,
        )
        self.assertIn("Ant-v5", prompt)
        self.assertIn("action_l2", prompt)
        self.assertIn("Current best reward expression", prompt)
        self.assertIn("Task context", prompt)
        self.assertIn("Environment source code excerpt", prompt)
        self.assertIn("original_reward = forward_reward + survive_reward - control_cost", prompt)

    def test_build_reward_prompt_includes_eureka_elites(self) -> None:
        prompt = build_reward_prompt(
            task="Ant-v5",
            reward_variables=["action_l2", "survive_reward", "x_velocity"],
            best_expression="x_velocity",
            best_score=10.0,
            elites=[{"name": "elite_0", "expression": "x_velocity", "score": 10.0}],
        )
        self.assertIn("EUREKA elite archive", prompt)
        self.assertIn("elite_0", prompt)

    def test_build_reward_prompt_prefers_evolution_feedback(self) -> None:
        prompt = build_reward_prompt(
            task="Ant-v5",
            reward_variables=["action_l2", "survive_reward", "x_velocity"],
            best_expression="x_velocity",
            best_score=10.0,
            elites=[{"name": "elite_0", "expression": "x_velocity", "score": 10.0}],
            evolution_feedback="ELITE name=elite_0 verified_return=10.0",
        )
        self.assertIn("EUREKA evolutionary feedback", prompt)
        self.assertIn("verified_return=10.0", prompt)

    def test_token_logprobs_matches_generated_tokens(self) -> None:
        import torch

        scores = (
            torch.tensor([[0.0, 1.0, 2.0]]),
            torch.tensor([[2.0, 0.0, 1.0]]),
        )
        logprobs = token_logprobs(scores, [2, 0])
        self.assertEqual(len(logprobs), 2)
        self.assertGreater(logprobs[0], logprobs[1] - 1.0)


if __name__ == "__main__":
    unittest.main()
