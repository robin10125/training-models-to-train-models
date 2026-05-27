from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rewards import RewardCandidate


@dataclass(frozen=True)
class CandidateResult:
    candidate: RewardCandidate
    mean_reward: float | None
    std_reward: float | None
    episode_rewards: list[float]
    timesteps: int
    seed: int
    task: str
    verified_reward_type: str = "true_env_return"
    rlvr_reward: float | None = None
    rlvr_reward_type: str | None = None
    status: str = "success"
    error: str | None = None
    elapsed_seconds: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MjwarpOptions:
    evaluator: str
    episode_steps: int
    training_episode_horizon: int
    policy_iterations: int
    ppo_horizon: int
    ppo_epochs: int
    ppo_minibatch_size: int
    ppo_learning_rate: float
    ppo_init_mode: str
    base_policy_checkpoint: str | None
    elite_frac: float
    rollout_mode: str
    verified_evaluator: str
    verification_steps: int
    verified_audit_gym: bool
    verified_audit_max_abs_diff: float | None
    reward_backend: str
    batch_candidates: bool
    cuda_graph: bool

    @classmethod
    def from_source(cls, source: Any) -> "MjwarpOptions":
        return cls(
            evaluator=source.mjwarp_evaluator,
            episode_steps=source.mjwarp_episode_steps,
            training_episode_horizon=source.mjwarp_training_episode_horizon,
            policy_iterations=source.mjwarp_policy_iterations,
            ppo_horizon=source.mjwarp_ppo_horizon,
            ppo_epochs=source.mjwarp_ppo_epochs,
            ppo_minibatch_size=source.mjwarp_ppo_minibatch_size,
            ppo_learning_rate=source.mjwarp_ppo_learning_rate,
            ppo_init_mode=source.mjwarp_ppo_init_mode,
            base_policy_checkpoint=source.mjwarp_base_policy_checkpoint,
            elite_frac=source.mjwarp_elite_frac,
            rollout_mode=source.mjwarp_rollout_mode,
            verified_evaluator=source.mjwarp_verified_evaluator,
            verification_steps=source.mjwarp_verification_steps,
            verified_audit_gym=source.mjwarp_verified_audit_gym,
            verified_audit_max_abs_diff=source.mjwarp_verified_audit_max_abs_diff,
            reward_backend=source.mjwarp_reward_backend,
            batch_candidates=source.mjwarp_batch_candidates,
            cuda_graph=source.mjwarp_cuda_graph,
        )

    def candidate_kwargs(self) -> dict[str, Any]:
        return {f"mjwarp_{key}": value for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class CandidateEvaluationConfig:
    task: str
    timesteps: int
    eval_episodes: int
    n_envs: int
    seed: int
    device: str
    sim_backend: str = "sb3"
    worlds_per_candidate: int = 4096
    mjwarp_evaluator: str = "ppo"
    mjwarp_episode_steps: int = 500
    mjwarp_training_episode_horizon: int = 1000
    mjwarp_policy_iterations: int = 96
    mjwarp_ppo_horizon: int = 32
    mjwarp_ppo_epochs: int = 4
    mjwarp_ppo_minibatch_size: int = 16_384
    mjwarp_ppo_learning_rate: float = 3.0e-4
    mjwarp_ppo_init_mode: str = "scratch"
    mjwarp_base_policy_checkpoint: str | None = None
    mjwarp_elite_frac: float = 0.1
    mjwarp_rollout_mode: str = "gpu"
    mjwarp_verified_evaluator: str = "mjwarp"
    mjwarp_verification_steps: int = 1000
    mjwarp_verified_audit_gym: bool = False
    mjwarp_verified_audit_max_abs_diff: float | None = None
    mjwarp_reward_backend: str = "eager"
    mjwarp_batch_candidates: bool = True
    mjwarp_cuda_graph: bool = True

    def mjwarp_options(self) -> MjwarpOptions:
        return MjwarpOptions.from_source(self)


@dataclass(frozen=True)
class RunConfig:
    task: str
    generations: int
    population: int
    eureka_elites: int
    timesteps: int
    eval_episodes: int
    n_envs: int
    seed: int
    device: str
    generator: str
    model_id: str
    adapter_path: str | None
    max_new_tokens: int
    temperature: float
    top_p: float
    load_in_4bit: bool
    sim_backend: str = "sb3"
    worlds_per_candidate: int = 4096
    mjwarp_evaluator: str = "ppo"
    mjwarp_episode_steps: int = 500
    mjwarp_training_episode_horizon: int = 1000
    mjwarp_policy_iterations: int = 96
    mjwarp_ppo_horizon: int = 32
    mjwarp_ppo_epochs: int = 4
    mjwarp_ppo_minibatch_size: int = 16_384
    mjwarp_ppo_learning_rate: float = 3.0e-4
    mjwarp_ppo_init_mode: str = "scratch"
    mjwarp_base_policy_checkpoint: str | None = None
    mjwarp_elite_frac: float = 0.1
    mjwarp_rollout_mode: str = "gpu"
    mjwarp_verified_evaluator: str = "mjwarp"
    mjwarp_verification_steps: int = 1000
    mjwarp_verified_audit_gym: bool = False
    mjwarp_verified_audit_max_abs_diff: float | None = None
    mjwarp_reward_backend: str = "eager"
    mjwarp_batch_candidates: bool = True
    mjwarp_cuda_graph: bool = True
    include_negative_rlvr_samples: bool = True
    negative_rlvr_margin: float = 1.0

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RunConfig":
        return cls(
            task=row["task"],
            generations=int(row["generations"]),
            population=int(row["population"]),
            eureka_elites=int(row.get("eureka_elites", 4)),
            timesteps=int(row["timesteps"]),
            eval_episodes=int(row["eval_episodes"]),
            n_envs=int(row["n_envs"]),
            seed=int(row["seed"]),
            device=row["device"],
            generator=row["generator"],
            model_id=row["model_id"],
            adapter_path=row.get("adapter_path"),
            max_new_tokens=int(row["max_new_tokens"]),
            temperature=float(row["temperature"]),
            top_p=float(row["top_p"]),
            load_in_4bit=bool(row["load_in_4bit"]),
            sim_backend=row.get("sim_backend", "sb3"),
            worlds_per_candidate=int(row.get("worlds_per_candidate", 4096)),
            mjwarp_evaluator=row.get("mjwarp_evaluator", "ppo"),
            mjwarp_episode_steps=int(row.get("mjwarp_episode_steps", 500)),
            mjwarp_training_episode_horizon=int(row.get("mjwarp_training_episode_horizon", 1000)),
            mjwarp_policy_iterations=int(row.get("mjwarp_policy_iterations", 96)),
            mjwarp_ppo_horizon=int(row.get("mjwarp_ppo_horizon", 32)),
            mjwarp_ppo_epochs=int(row.get("mjwarp_ppo_epochs", 4)),
            mjwarp_ppo_minibatch_size=int(row.get("mjwarp_ppo_minibatch_size", 16_384)),
            mjwarp_ppo_learning_rate=float(row.get("mjwarp_ppo_learning_rate", 3.0e-4)),
            mjwarp_ppo_init_mode=row.get("mjwarp_ppo_init_mode", "scratch"),
            mjwarp_base_policy_checkpoint=row.get("mjwarp_base_policy_checkpoint"),
            mjwarp_elite_frac=float(row.get("mjwarp_elite_frac", 0.1)),
            mjwarp_rollout_mode=row.get("mjwarp_rollout_mode", "gpu"),
            mjwarp_verified_evaluator=row.get("mjwarp_verified_evaluator", "mjwarp"),
            mjwarp_verification_steps=int(row.get("mjwarp_verification_steps", 1000)),
            mjwarp_verified_audit_gym=bool(row.get("mjwarp_verified_audit_gym", False)),
            mjwarp_verified_audit_max_abs_diff=(
                None
                if row.get("mjwarp_verified_audit_max_abs_diff") is None
                else float(row["mjwarp_verified_audit_max_abs_diff"])
            ),
            mjwarp_reward_backend=row.get("mjwarp_reward_backend", "eager"),
            mjwarp_batch_candidates=bool(row.get("mjwarp_batch_candidates", True)),
            mjwarp_cuda_graph=bool(row.get("mjwarp_cuda_graph", True)),
            include_negative_rlvr_samples=bool(row.get("include_negative_rlvr_samples", True)),
            negative_rlvr_margin=float(row.get("negative_rlvr_margin", 1.0)),
        )

    def mjwarp_options(self) -> MjwarpOptions:
        return MjwarpOptions.from_source(self)

    def evaluation_config(self, seed: int) -> CandidateEvaluationConfig:
        return CandidateEvaluationConfig(
            task=self.task,
            timesteps=self.timesteps,
            eval_episodes=self.eval_episodes,
            n_envs=self.n_envs,
            seed=seed,
            device=self.device,
            sim_backend=self.sim_backend,
            worlds_per_candidate=self.worlds_per_candidate,
            **self.mjwarp_options().candidate_kwargs(),
        )
