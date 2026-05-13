from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eureka_lite.pipeline import FullPipelineConfig, run_full_pipeline
from eureka_lite.search import CandidateResult
from eureka_lite.generators import initial_population


class PipelineTests(unittest.TestCase):
    def test_run_full_pipeline_collects_then_trains(self) -> None:
        candidate = initial_population("Ant-v5", 1, __import__("random").Random(7))[0]
        result = CandidateResult(
            candidate=candidate,
            mean_reward=1.0,
            std_reward=0.0,
            episode_rewards=[1.0],
            timesteps=10,
            seed=7,
            task="Ant-v5",
        )
        with tempfile.TemporaryDirectory() as tmp:
            collection_dir = Path(tmp) / "collection"
            adapter_dir = Path(tmp) / "adapter"

            def fake_run_search(**kwargs):
                collection_dir.mkdir(parents=True, exist_ok=True)
                (collection_dir / "rlvr_records.jsonl").write_text(
                    json.dumps(
                        {
                            "status": "success",
                            "verified_reward": 1.0,
                            "prompt": "p",
                            "completion": "c",
                            "completion_token_ids": [1],
                            "old_logprobs": [-0.1],
                            "prompt_id": "p",
                            "generator_checkpoint": "m",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return [result]

            with patch("eureka_lite.pipeline.run_search", side_effect=fake_run_search) as run_search_mock:
                with patch("eureka_lite.pipeline.train_rlvr", return_value={"final_loss": 0.5}) as train_mock:
                    summary = run_full_pipeline(
                        FullPipelineConfig(
                            task="Ant-v5",
                            model_id="model",
                            collection_output_dir=collection_dir.as_posix(),
                            adapter_output_dir=adapter_dir.as_posix(),
                            population=16,
                            generations=1,
                            worlds_per_candidate=4096,
                            mjwarp_episode_steps=500,
                            mjwarp_policy_iterations=4,
                            mjwarp_elite_frac=0.1,
                            eval_episodes=5,
                            seed=7,
                            device="cuda",
                            max_new_tokens=256,
                            temperature=0.7,
                            top_p=0.95,
                            load_in_4bit=True,
                            trainer_algorithm="grpo",
                            trainer_epochs=1,
                            trainer_batch_size=1,
                            trainer_learning_rate=5e-5,
                            trainer_max_length=1024,
                            trainer_max_grad_norm=1.0,
                            trainer_lora_r=16,
                            trainer_lora_alpha=32,
                            trainer_lora_dropout=0.05,
                            trainer_clip_epsilon=0.2,
                            trainer_beta_kl=0.01,
                            overwrite_collection=False,
                            force_train=False,
                        )
                    )

        self.assertEqual(summary["collection_results"], 1)
        self.assertEqual(summary["trainer_status"], "trained")
        self.assertEqual(run_search_mock.call_args.kwargs["sim_backend"], "mjwarp")
        self.assertEqual(run_search_mock.call_args.kwargs["worlds_per_candidate"], 4096)
        self.assertEqual(train_mock.call_args.args[0].algorithm, "grpo")


if __name__ == "__main__":
    unittest.main()
