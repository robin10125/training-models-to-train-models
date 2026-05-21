from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eureka_lite.generators import initial_population
from eureka_lite.search import (
    CandidateResult,
    RunConfig,
    candidate_result_from_dict,
    candidate_result_to_dict,
    prepare_output_dir,
    run_search,
    save_checkpoint,
    to_rlvr_record,
)


class CheckpointingTests(unittest.TestCase):
    def test_candidate_result_round_trips_through_dict(self) -> None:
        candidate = initial_population("Ant-v5", 1, __import__("random").Random(7))[0]
        result = CandidateResult(
            candidate=candidate,
            mean_reward=1.5,
            std_reward=0.25,
            episode_rewards=[1.25, 1.75],
            timesteps=100,
            seed=7,
            task="Ant-v5",
            elapsed_seconds=2.0,
        )
        restored = candidate_result_from_dict(candidate_result_to_dict(result))
        self.assertEqual(restored.candidate.name, candidate.name)
        self.assertEqual(restored.mean_reward, 1.5)
        self.assertEqual(restored.episode_rewards, [1.25, 1.75])

    def test_checkpoint_file_contains_resume_state(self) -> None:
        candidate = initial_population("Ant-v5", 1, __import__("random").Random(7))[0]
        result = CandidateResult(
            candidate=candidate,
            mean_reward=None,
            std_reward=None,
            episode_rewards=[],
            timesteps=0,
            seed=7,
            task="Ant-v5",
            status="generated_only",
        )
        config = RunConfig(
            task="Ant-v5",
            generations=1,
            population=1,
            eureka_elites=1,
            timesteps=0,
            eval_episodes=0,
            n_envs=1,
            seed=7,
            device="cpu",
            generator="mock",
            model_id="mock",
            adapter_path=None,
            max_new_tokens=1,
            temperature=0.1,
            top_p=0.9,
            load_in_4bit=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prepare_output_dir(output_dir, config, resume=False, overwrite=False)
            save_checkpoint(
                output_dir,
                config,
                results=[result],
                next_generation=1,
                next_candidates=[],
                best_expression=candidate.expression,
                best_score=None,
            )
            payload = json.loads((output_dir / "checkpoint.json").read_text())
            self.assertEqual(payload["next_generation"], 1)
            self.assertEqual(payload["results"][0]["status"], "generated_only")

    def test_rlvr_record_includes_error_and_elapsed_seconds(self) -> None:
        candidate = initial_population("Ant-v5", 1, __import__("random").Random(7))[0]
        result = CandidateResult(
            candidate=candidate,
            mean_reward=None,
            std_reward=None,
            episode_rewards=[],
            timesteps=0,
            seed=7,
            task="Ant-v5",
            status="failed",
            error="boom",
            elapsed_seconds=1.0,
        )
        record = to_rlvr_record(result)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "boom")
        self.assertEqual(record["elapsed_seconds"], 1.0)

    def test_search_records_eureka_lineage_and_rank_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            results = run_search(
                task="Ant-v5",
                generations=2,
                population=3,
                eureka_elites=2,
                timesteps=0,
                eval_episodes=0,
                n_envs=1,
                seed=7,
                device="cpu",
                output_dir=output_dir,
                generator="mock",
            )
            records = [
                json.loads(line)
                for line in (output_dir / "rlvr_records.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(results), 6)
        self.assertTrue(any(record["eureka_parent_names"] for record in records))
        self.assertTrue(any(record["eureka_elite_names"] for record in records))
        self.assertTrue(
            any(
                record["metadata"] is not None and record["metadata"].get("eureka_selected_elite")
                for record in records
            )
        )

    def test_resume_completed_run_is_noop(self) -> None:
        candidate = initial_population("Ant-v5", 1, __import__("random").Random(7))[0]
        result = CandidateResult(
            candidate=candidate,
            mean_reward=None,
            std_reward=None,
            episode_rewards=[],
            timesteps=0,
            seed=7,
            task="Ant-v5",
            status="generated_only",
        )
        config = RunConfig(
            task="Ant-v5",
            generations=1,
            population=1,
            eureka_elites=1,
            timesteps=0,
            eval_episodes=0,
            n_envs=1,
            seed=7,
            device="cpu",
            generator="mock",
            model_id="mock",
            adapter_path=None,
            max_new_tokens=1,
            temperature=0.1,
            top_p=0.9,
            load_in_4bit=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            prepare_output_dir(output_dir, config, resume=False, overwrite=False)
            save_checkpoint(
                output_dir,
                config,
                results=[result],
                next_generation=1,
                next_candidates=[],
                best_expression=candidate.expression,
                best_score=None,
            )
            resumed = run_search(
                task="CartPole-v1",
                generations=99,
                population=99,
                timesteps=99,
                eval_episodes=99,
                n_envs=99,
                seed=99,
                device="auto",
                output_dir=output_dir,
                generator="mock",
                resume=True,
            )
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed[0].task, "Ant-v5")


if __name__ == "__main__":
    unittest.main()
