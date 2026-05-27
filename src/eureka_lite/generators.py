from __future__ import annotations

import random

from .adapters import ANT_TASK
from .rewards import RewardCandidate


MOCK_GENERATOR_TYPE = "mock"
MOCK_GENERATOR_CHECKPOINT = "mock-ant-v1"


def ant_expression(weights: dict[str, float]) -> str:
    return (
        f"{weights['forward']:.4f} * x_velocity"
        f" + {weights['survive']:.4f} * survive_reward"
        f" - {weights['lateral']:.4f} * abs(y_velocity)"
        f" - {weights['control']:.4f} * action_l2"
        f" - {weights['height']:.4f} * abs(torso_z - 0.55)"
        f" - ({weights['failure']:.4f} if terminated else 0.0)"
    )


def initial_population(task: str, population: int, rng: random.Random) -> list[RewardCandidate]:
    if task != ANT_TASK:
        raise ValueError(f"Unsupported task for mock generator: {task}")
    return _ant_initial_population(population, rng)


def mutate_candidate(parent: RewardCandidate, index: int, generation: int, rng: random.Random) -> RewardCandidate:
    if parent.task != ANT_TASK:
        raise ValueError(f"Unsupported task for mock generator: {parent.task}")

    weights = dict(parent.weights)
    for key in weights:
        weights[key] = max(0.0, weights[key] * rng.uniform(0.65, 1.45) + rng.uniform(-0.02, 0.02))
    expression = ant_expression(weights)
    return _candidate(
        name=f"gen{generation}_mut{index}_from_{parent.name}",
        task=parent.task,
        prompt_id=parent.prompt_id,
        expression=expression,
        weights=weights,
        generation=generation,
        eureka_role="mock_mutation",
        eureka_parent_names=[parent.name],
        eureka_parent_expressions=[parent.expression],
    )


def _ant_initial_population(population: int, rng: random.Random) -> list[RewardCandidate]:
    baseline_weights = {
        "forward": 1.0,
        "survive": 1.0,
        "lateral": 0.2,
        "control": 0.03,
        "height": 0.3,
        "failure": 2.0,
    }
    candidates = [
        _candidate(
            name="baseline_forward_survive",
            task=ANT_TASK,
            prompt_id="ant_reward_design_v1",
            expression=ant_expression(baseline_weights),
            weights=baseline_weights,
            generation=0,
        )
    ]
    for index in range(max(0, population - 1)):
        weights = {
            "forward": rng.uniform(0.4, 2.0),
            "survive": rng.uniform(0.2, 1.5),
            "lateral": rng.uniform(0.0, 0.8),
            "control": rng.uniform(0.0, 0.12),
            "height": rng.uniform(0.0, 1.0),
            "failure": rng.uniform(0.5, 5.0),
        }
        candidates.append(
            _candidate(
                name=f"gen0_random{index}",
                task=ANT_TASK,
                prompt_id="ant_reward_design_v1",
                expression=ant_expression(weights),
                weights=weights,
                generation=0,
            )
        )
    return candidates


def _candidate(
    *,
    name: str,
    task: str,
    prompt_id: str,
    expression: str,
    weights: dict[str, float],
    generation: int,
    eureka_role: str = "initial",
    eureka_parent_names: list[str] | None = None,
    eureka_parent_expressions: list[str] | None = None,
    eureka_parent_scores: list[float | None] | None = None,
    eureka_elite_names: list[str] | None = None,
    eureka_elite_expressions: list[str] | None = None,
    eureka_elite_scores: list[float | None] | None = None,
    eureka_feedback: str | None = None,
) -> RewardCandidate:
    prompt = (
        f"Design a dense reward expression for {task}. "
        "The expression will train a PPO policy, and the candidate will be scored "
        "only by verified target-environment return during evaluation."
    )
    return RewardCandidate(
        name=name,
        task=task,
        prompt_id=prompt_id,
        prompt=prompt,
        expression=expression,
        weights=weights,
        generation=generation,
        generator_type=MOCK_GENERATOR_TYPE,
        generator_checkpoint=MOCK_GENERATOR_CHECKPOINT,
        eureka_role=eureka_role,
        eureka_parent_names=eureka_parent_names,
        eureka_parent_expressions=eureka_parent_expressions,
        eureka_parent_scores=eureka_parent_scores,
        eureka_elite_names=eureka_elite_names,
        eureka_elite_expressions=eureka_elite_expressions,
        eureka_elite_scores=eureka_elite_scores,
        eureka_feedback=eureka_feedback,
    )
