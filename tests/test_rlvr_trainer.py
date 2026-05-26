from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from eureka_lite.rlvr_trainer import (
    RlvrDataset,
    RlvrTrainingExample,
    align_old_logprobs,
    collate_batch,
    grpo_clipped_loss,
    load_grpo_examples,
    load_rlvr_examples,
    weighted_completion_loss,
)


class RlvrTrainerTests(unittest.TestCase):
    def test_grpo_dataset_uses_sampled_token_ids_and_rejects_context_truncation(self) -> None:
        class Tokenizer:
            eos_token_id = 9

            def __call__(self, text, add_special_tokens=False):
                del add_special_tokens
                return type("Result", (), {"input_ids": [1] * len(text)})()

        example = RlvrTrainingExample(
            prompt="p",
            completion="decoded-text",
            reward=1.0,
            advantage=1.0,
            source="test",
            completion_token_ids=[7, 8],
            old_logprobs=[-0.1, -0.2],
        )
        row = RlvrDataset([example], Tokenizer(), max_length=8)[0]
        self.assertEqual(row["input_ids"], [1, 7, 8, 9])
        with self.assertRaisesRegex(ValueError, "trainer_max_length"):
            RlvrDataset([example], Tokenizer(), max_length=2)[0]

    def test_load_rlvr_examples_filters_and_normalizes_advantages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            rows = [
                {"status": "success", "verified_reward": 10.0, "prompt": "p1", "completion": "c1"},
                {"status": "success", "verified_reward": 20.0, "prompt": "p2", "completion": "c2"},
                {"status": "failed", "verified_reward": 999.0, "prompt": "bad", "completion": "bad"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            examples = load_rlvr_examples(path)
        self.assertEqual(len(examples), 2)
        self.assertLess(examples[0].advantage, 0)
        self.assertGreater(examples[1].advantage, 0)

    def test_collate_batch_pads_inputs_and_labels(self) -> None:
        batch = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2], "advantage": 1.0},
            {"input_ids": [3], "attention_mask": [1], "labels": [3], "advantage": -1.0},
        ]
        collated = collate_batch(batch, pad_token_id=0)
        self.assertEqual(collated["input_ids"].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(collated["labels"].tolist(), [[-100, 2], [3, -100]])
        self.assertEqual(collated["advantages"].shape[0], 2)

    def test_weighted_completion_loss_is_finite(self) -> None:
        logits = torch.randn(2, 4, 8)
        labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 5]])
        advantages = torch.tensor([1.0, -1.0])
        loss = weighted_completion_loss(logits, labels, advantages)
        self.assertTrue(torch.isfinite(loss))

    def test_load_grpo_examples_uses_group_relative_advantages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            base = {
                "status": "success",
                "prompt": "same prompt",
                "prompt_id": "p",
                "generator_checkpoint": "m",
                "completion_token_ids": [1, 2],
                "old_logprobs": [-0.1, -0.2],
            }
            rows = [
                {**base, "verified_reward": 10.0, "completion": "low"},
                {**base, "verified_reward": 20.0, "completion": "high"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            examples = load_grpo_examples(path)
        self.assertEqual(len(examples), 2)
        self.assertLess(examples[0].advantage, 0)
        self.assertGreater(examples[1].advantage, 0)
        self.assertEqual(examples[0].old_logprobs, [-0.1, -0.2])

    def test_load_grpo_examples_includes_negative_rlvr_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            base = {
                "prompt": "same prompt",
                "prompt_id": "p",
                "generator_checkpoint": "m",
                "completion_token_ids": [1, 2],
                "old_logprobs": [-0.1, -0.2],
            }
            rows = [
                {**base, "status": "success", "verified_reward": 10.0, "completion": "valid"},
                {
                    **base,
                    "status": "invalid_completion",
                    "verified_reward": None,
                    "rlvr_reward": 5.0,
                    "completion": "invalid",
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            examples = load_grpo_examples(path)
        self.assertEqual(len(examples), 2)
        by_completion = {example.completion: example for example in examples}
        self.assertLess(by_completion["invalid"].advantage, 0)
        self.assertGreater(by_completion["valid"].advantage, 0)

    def test_align_old_logprobs_matches_completion_labels(self) -> None:
        labels = [-100, -100, 10, 11]
        aligned = align_old_logprobs(labels, [-0.5, -0.6])
        self.assertEqual(aligned, [None, None, -0.5, -0.6])

    def test_grpo_clipped_loss_is_finite(self) -> None:
        logits = torch.randn(2, 4, 8)
        labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 5]])
        old_logprobs = torch.tensor([[-1e9, -0.1, -0.2, -0.3], [-1e9, -1e9, -0.4, -0.5]])
        advantages = torch.tensor([1.0, -1.0])
        loss = grpo_clipped_loss(
            logits,
            labels,
            advantages,
            old_logprobs,
            clip_epsilon=0.2,
            beta_kl=0.01,
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
