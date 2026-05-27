from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from eureka_lite.generators import initial_population
from eureka_lite.search import (
    CandidateEvaluationConfig,
    CandidateResult,
    RunConfig,
    candidate_result_from_dict,
    candidate_result_to_dict,
    candidates_can_be_batched,
    prepare_output_dir,
    assign_failed_evaluation_penalties,
    negative_reward_for_generation,
    rejected_candidates_to_results,
    run_search,
    save_checkpoint,
    to_rlvr_record,
    verified_reward_type_for_evaluator,
)


class CheckpointingTests(unittest.TestCase):
    def test_verified_reward_type_names_target_and_transfer_domains(self) -> None:
        self.assertEqual(verified_reward_type_for_evaluator("mjwarp"), "mjwarp_ant_return")
        self.assertEqual(verified_reward_type_for_evaluator("gym"), "gym_ant_v5_return")

    def test_gpu_mjwarp_ppo_candidates_are_batched_by_default(self) -> None:
        candidates = initial_population("Ant-v5", 2, __import__("random").Random(7))
        config = CandidateEvaluationConfig(
            task="Ant-v5",
            timesteps=1,
            eval_episodes=1,
            n_envs=1,
            seed=7,
            device="cuda",
            sim_backend="mjwarp",
        )
        self.assertTrue(candidates_can_be_batched(candidates, config))
        self.assertFalse(candidates_can_be_batched(candidates, config.__class__(**{**config.__dict__, "mjwarp_batch_candidates": False})))

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
                evolution_feedback="reflection",
            )
            payload = json.loads((output_dir / "checkpoint.json").read_text())
            self.assertEqual(payload["next_generation"], 1)
            self.assertEqual(payload["results"][0]["status"], "generated_only")
            self.assertEqual(payload["evolution_feedback"], "reflection")

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

    def test_failed_and_invalid_candidates_receive_separate_rlvr_penalties(self) -> None:
        candidate = initial_population("Ant-v5", 1, __import__("random").Random(7))[0]
        success = CandidateResult(
            candidate=candidate,
            mean_reward=10.0,
            std_reward=0.0,
            episode_rewards=[10.0],
            timesteps=1,
            seed=7,
            task="Ant-v5",
        )
        failed = CandidateResult(
            candidate=candidate,
            mean_reward=None,
            std_reward=None,
            episode_rewards=[],
            timesteps=1,
            seed=8,
            task="Ant-v5",
            status="failed",
            error="crash",
        )
        penalty = negative_reward_for_generation([success, failed], margin=2.0)
        penalized = assign_failed_evaluation_penalties([failed], penalty)[0]
        rejected = rejected_candidates_to_results(
            [(candidate, "ValueError: invalid")],
            task="Ant-v5",
            seed=9,
            penalty=penalty,
        )[0]
        self.assertEqual(penalty, 8.0)
        self.assertEqual(to_rlvr_record(penalized)["verified_reward"], None)
        self.assertEqual(to_rlvr_record(penalized)["rlvr_reward"], 8.0)
        self.assertEqual(to_rlvr_record(rejected)["status"], "invalid_completion")
        self.assertEqual(to_rlvr_record(rejected)["rlvr_reward_type"], "invalid_completion_penalty")

    def test_search_records_eureka_lineage_and_rank_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            results = run_search(
                RunConfig(
                    task="Ant-v5",
                    generations=2,
                    population=3,
                    eureka_elites=2,
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
                    load_in_4bit=False,
                ),
                output_dir=output_dir,
            )
            records = [
                json.loads(line)
                for line in (output_dir / "rlvr_records.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(results), 6)
        self.assertEqual({result.seed for result in results if result.candidate.generation == 0}, {7})
        self.assertEqual({result.seed for result in results if result.candidate.generation == 1}, {107})
        self.assertTrue(any(record["eureka_parent_names"] for record in records))
        self.assertTrue(any(record["eureka_elite_names"] for record in records))
        self.assertTrue(any(record["eureka_feedback"] for record in records))
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
                RunConfig(
                    task="Unknown-v0",
                    generations=99,
                    population=99,
                    eureka_elites=1,
                    timesteps=99,
                    eval_episodes=99,
                    n_envs=99,
                    seed=99,
                    device="auto",
                    generator="mock",
                    model_id="mock",
                    adapter_path=None,
                    max_new_tokens=1,
                    temperature=0.1,
                    top_p=0.9,
                    load_in_4bit=False,
                ),
                output_dir=output_dir,
                resume=True,
            )
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed[0].task, "Ant-v5")

    def test_hf_resume_after_generation_boundary_generates_next_generation(self) -> None:
        generations_requested: list[int] = []

        class FakeGenerator:
            def __init__(self, _config) -> None:
                self._rejections = []

            def generate_population(self, *, task, population, generation, **_kwargs):
                generations_requested.append(generation)
                base = initial_population(task, population, __import__("random").Random(7))
                return [
                    replace(candidate, name=f"hf_g{generation}_{index}", generation=generation, generator_type="hf")
                    for index, candidate in enumerate(base)
                ]

            def drain_rejected_candidates(self):
                return self._rejections

        config = RunConfig(
            task="Ant-v5",
            generations=2,
            population=1,
            eureka_elites=1,
            timesteps=0,
            eval_episodes=0,
            n_envs=1,
            seed=7,
            device="cpu",
            generator="hf",
            model_id="mock",
            adapter_path=None,
            max_new_tokens=1,
            temperature=0.1,
            top_p=0.9,
            load_in_4bit=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch("eureka_lite.search.HfRewardGenerator", FakeGenerator):
                with patch("eureka_lite.search.pause_requested", side_effect=[False, True]):
                    run_search(config, output_dir=output_dir, pause_path=output_dir / "PAUSE")
                results = run_search(config, output_dir=output_dir, resume=True)

        self.assertEqual(generations_requested, [0, 1])
        self.assertEqual({result.candidate.generation for result in results}, {0, 1})

    def test_batched_evaluation_publishes_results_once_after_finalization(self) -> None:
        config = RunConfig(
            task="Ant-v5",
            generations=1,
            population=2,
            eureka_elites=1,
            timesteps=1,
            eval_episodes=1,
            n_envs=1,
            seed=7,
            device="cuda",
            generator="mock",
            model_id="mock",
            adapter_path=None,
            max_new_tokens=1,
            temperature=0.1,
            top_p=0.9,
            load_in_4bit=False,
            sim_backend="mjwarp",
        )

        def fake_evaluate(candidates, evaluation_config):
            return [
                CandidateResult(
                    candidate=candidate,
                    mean_reward=float(index),
                    std_reward=0.0,
                    episode_rewards=[float(index)],
                    timesteps=1,
                    seed=evaluation_config.seed,
                    task=evaluation_config.task,
                )
                for index, candidate in enumerate(candidates)
            ]

        with tempfile.TemporaryDirectory() as tmp:
            with patch("eureka_lite.search.run_candidates_safely", side_effect=fake_evaluate):
                with patch("eureka_lite.search.write_results") as write_results_mock:
                    run_search(config, output_dir=Path(tmp))
        write_results_mock.assert_called_once()

    def test_paused_partial_generation_does_not_publish_unfinalized_rlvr_records(self) -> None:
        config = RunConfig(
            task="Ant-v5",
            generations=1,
            population=2,
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
            load_in_4bit=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch("eureka_lite.search.pause_requested", return_value=True):
                run_search(config, output_dir=output_dir, pause_path=output_dir / "PAUSE")
            checkpoint = json.loads((output_dir / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(len(checkpoint["search_state"]["generation"]["raw_results"]), 1)
            self.assertFalse((output_dir / "rlvr_records.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
    candidates_can_be_batched,
