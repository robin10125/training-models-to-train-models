from __future__ import annotations

import random

from .rewards import RewardCandidate


MOCK_GENERATOR_TYPE = "mock"
MOCK_GENERATOR_CHECKPOINT = "mock-ant-v1"


def cartpole_expression(weights: dict[str, float]) -> str:
    return (
        f"{weights['alive']:.4f}"
        f" - {weights['x']:.4f} * abs(x) / 2.4"
        f" - {weights['theta']:.4f} * abs(theta) / 0.2095"
        f" - {weights['x_dot']:.4f} * abs(x_dot) / 3.0"
        f" - {weights['theta_dot']:.4f} * abs(theta_dot) / 3.5"
        f" - ({weights['failure']:.4f} if terminated else 0.0)"
    )


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
    if task == "CartPole-v1":
        return _cartpole_initial_population(population, rng)
    if task == "Ant-v5":
        return _ant_initial_population(population, rng)
    raise ValueError(f"Unsupported task for mock generator: {task}")


def mutate_candidate(parent: RewardCandidate, index: int, generation: int, rng: random.Random) -> RewardCandidate:
    if parent.task == "CartPole-v1":
        expression_builder = cartpole_expression
        jitter = 0.05
    elif parent.task == "Ant-v5":
        expression_builder = ant_expression
        jitter = 0.02
    else:
        raise ValueError(f"Unsupported task for mock generator: {parent.task}")

    weights = dict(parent.weights)
    for key in weights:
        weights[key] = max(0.0, weights[key] * rng.uniform(0.65, 1.45) + rng.uniform(-jitter, jitter))
    expression = expression_builder(weights)
    return _candidate(
        name=f"gen{generation}_mut{index}_from_{parent.name}",
        task=parent.task,
        prompt_id=parent.prompt_id,
        expression=expression,
        weights=weights,
        generation=generation,
    )


def _cartpole_initial_population(population: int, rng: random.Random) -> list[RewardCandidate]:
    baseline_weights = {
        "alive": 1.0,
        "x": 0.6,
        "theta": 2.4,
        "x_dot": 0.05,
        "theta_dot": 0.15,
        "failure": 4.0,
    }
    candidates = [
        _candidate(
            name="baseline_angle_position",
            task="CartPole-v1",
            prompt_id="cartpole_reward_design_v1",
            expression=cartpole_expression(baseline_weights),
            weights=baseline_weights,
            generation=0,
        )
    ]
    for index in range(max(0, population - 1)):
        weights = {
            "alive": rng.uniform(0.7, 1.4),
            "x": rng.uniform(0.1, 2.0),
            "theta": rng.uniform(0.5, 4.0),
            "x_dot": rng.uniform(0.0, 0.5),
            "theta_dot": rng.uniform(0.0, 0.8),
            "failure": rng.uniform(1.0, 8.0),
        }
        candidates.append(
            _candidate(
                name=f"gen0_random{index}",
                task="CartPole-v1",
                prompt_id="cartpole_reward_design_v1",
                expression=cartpole_expression(weights),
                weights=weights,
                generation=0,
            )
        )
    return candidates


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
            task="Ant-v5",
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
                task="Ant-v5",
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
) -> RewardCandidate:
    prompt = (
        f"Design a dense reward expression for {task}. "
        "The expression will train a PPO policy, and the candidate will be scored "
        "only by true environment return during evaluation."
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
    )

