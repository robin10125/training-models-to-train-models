from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .adapters import get_adapter
from .rewards import ALLOWED_FUNCS, RewardCandidate, RewardExpression


@dataclass(frozen=True)
class MjwarpEvaluatorConfig:
    task: str = "Ant-v5"
    evaluator: str = "ppo"
    worlds_per_candidate: int = 4096
    episode_steps: int = 500
    policy_iterations: int = 4
    ppo_horizon: int = 32
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 16_384
    ppo_learning_rate: float = 3.0e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_value_coef: float = 2.0
    ppo_entropy_coef: float = 0.0
    ppo_max_grad_norm: float = 1.0
    elite_frac: float = 0.1
    init_std: float = 0.35
    min_std: float = 0.03
    seed: int = 7
    device: str = "cuda:0"
    eval_episodes: int = 5


def train_and_evaluate_mjwarp(candidate: RewardCandidate, config: MjwarpEvaluatorConfig) -> dict[str, Any]:
    if config.task != "Ant-v5":
        raise ValueError("The MJWarp evaluator currently supports only Ant-v5")
    if config.worlds_per_candidate < 2:
        raise ValueError("worlds_per_candidate must be at least 2")
    if config.episode_steps < 1:
        raise ValueError("episode_steps must be at least 1")
    if config.policy_iterations < 1:
        raise ValueError("policy_iterations must be at least 1")
    if config.evaluator not in {"ppo", "search"}:
        raise ValueError("evaluator must be one of: ppo, search")
    if config.ppo_horizon < 1:
        raise ValueError("ppo_horizon must be at least 1")
    if not 0.0 < config.elite_frac <= 1.0:
        raise ValueError("elite_frac must be in (0, 1]")

    try:
        import mujoco_warp as mjw
        import warp as wp
    except ImportError as exc:
        raise RuntimeError(
            "The MJWarp evaluator requires optional GPU dependencies. Install with "
            "`pip install -e '.[mjwarp]'`."
        ) from exc

    started_at = time.monotonic()
    adapter = get_adapter(config.task)
    reward_expression = VectorizedRewardExpression(candidate.expression, adapter.reward_variables)
    rng = np.random.default_rng(config.seed)

    env = gym.make(config.task)
    try:
        mjm = env.unwrapped.model
        dt = float(mjm.opt.timestep * env.unwrapped.frame_skip)
        action_dim = int(mjm.nu)
        obs_dim = int(mjm.nq - 2 + mjm.nv)
        wp.init()
        with wp.ScopedDevice(config.device):
            model = mjw.put_model(mjm)
            data = mjw.make_data(mjm, nworld=config.worlds_per_candidate)
            if config.evaluator == "ppo":
                policy, best_shaped_return, iteration_summaries = train_ppo_policy(
                    model=model,
                    data=data,
                    mjm=mjm,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    reward_expression=reward_expression,
                    config=config,
                    dt=dt,
                    wp=wp,
                    mjw=mjw,
                )
                eval_policy = ("ppo", policy)
            else:
                best_params, best_shaped_return, iteration_summaries = train_search_policy(
                    model=model,
                    data=data,
                    mjm=mjm,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    reward_expression=reward_expression,
                    config=config,
                    dt=dt,
                    rng=rng,
                    wp=wp,
                    mjw=mjw,
                )
                eval_policy = ("linear", best_params)
    finally:
        env.close()

    eval_returns = evaluate_policy_in_gym(
        task=config.task,
        policy=eval_policy,
        obs_dim=obs_dim,
        action_dim=action_dim,
        eval_episodes=config.eval_episodes,
        seed=config.seed + 10_000,
    )
    return {
        "mean_reward": float(np.mean(eval_returns)),
        "std_reward": float(np.std(eval_returns)),
        "episode_rewards": eval_returns,
        "elapsed_seconds": time.monotonic() - started_at,
        "metadata": {
            "sim_backend": "mjwarp",
            "mjwarp_evaluator": config.evaluator,
            "worlds_per_candidate": config.worlds_per_candidate,
            "episode_steps": config.episode_steps,
            "policy_iterations": config.policy_iterations,
            "training_world_steps": config.worlds_per_candidate * config.episode_steps * config.policy_iterations,
            "ppo_horizon": config.ppo_horizon,
            "ppo_epochs": config.ppo_epochs,
            "ppo_minibatch_size": config.ppo_minibatch_size,
            "ppo_hidden_sizes": [256, 128, 64],
            "elite_frac": config.elite_frac,
            "best_shaped_return": best_shaped_return,
            "iteration_summaries": iteration_summaries,
        },
    }


def rollout_policy_population(
    *,
    model: Any,
    data: Any,
    mjm: Any,
    params: np.ndarray,
    reward_expression: "VectorizedRewardExpression",
    episode_steps: int,
    dt: float,
    device: str,
    wp: Any,
    mjw: Any,
) -> tuple[np.ndarray, np.ndarray]:
    worlds, param_dim = params.shape
    action_dim = int(mjm.nu)
    obs_dim = int(param_dim // action_dim - 1)
    weights = params[:, : obs_dim * action_dim].reshape(worlds, obs_dim, action_dim)
    biases = params[:, obs_dim * action_dim :]
    shaped_returns = np.zeros(worlds, dtype=np.float32)
    true_returns = np.zeros(worlds, dtype=np.float32)
    terminated = np.zeros(worlds, dtype=bool)

    mjw.reset_data(model, data)
    wp.synchronize()
    for _step in range(episode_steps):
        qpos_before = np.asarray(data.qpos.numpy(), dtype=np.float32)
        qvel_before = np.asarray(data.qvel.numpy(), dtype=np.float32)
        obs = ant_policy_obs(qpos_before, qvel_before)
        action = np.tanh(np.einsum("wo,woa->wa", obs, weights) + biases).astype(np.float32)
        wp.copy(data.ctrl, wp.array(action, dtype=wp.float32, device=device))
        mjw.step(model, data)
        wp.synchronize()

        qpos_after = np.asarray(data.qpos.numpy(), dtype=np.float32)
        qvel_after = np.asarray(data.qvel.numpy(), dtype=np.float32)
        x_velocity = (qpos_after[:, 0] - qpos_before[:, 0]) / dt
        y_velocity = (qpos_after[:, 1] - qpos_before[:, 1]) / dt
        torso_z = qpos_after[:, 2]
        healthy = np.isfinite(qpos_after).all(axis=1) & np.isfinite(qvel_after).all(axis=1)
        healthy &= (torso_z >= 0.2) & (torso_z <= 1.0)
        just_terminated = ~terminated & ~healthy
        active = ~terminated
        terminated |= just_terminated

        action_l2 = np.square(action).sum(axis=1)
        forward_reward = x_velocity
        control_cost = 0.5 * 1.0e-2 * action_l2
        survive_reward = np.where(healthy, 1.0, 0.0)
        original_reward = forward_reward + survive_reward - control_cost
        context = {
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
            "control_cost": control_cost,
            "survive_reward": survive_reward,
            "torso_z": torso_z,
            "action_l2": action_l2,
            "original_reward": original_reward,
            "healthy": healthy,
            "terminated": just_terminated,
            "truncated": np.zeros(worlds, dtype=bool),
        }
        shaped_reward = reward_expression(context)
        shaped_returns += np.where(active, shaped_reward, 0.0).astype(np.float32)
        true_returns += np.where(active, original_reward, 0.0).astype(np.float32)
        if terminated.all():
            break
    return shaped_returns, true_returns


def train_search_policy(
    *,
    model: Any,
    data: Any,
    mjm: Any,
    obs_dim: int,
    action_dim: int,
    reward_expression: "VectorizedRewardExpression",
    config: MjwarpEvaluatorConfig,
    dt: float,
    rng: np.random.Generator,
    wp: Any,
    mjw: Any,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    param_dim = obs_dim * action_dim + action_dim
    mean = np.zeros(param_dim, dtype=np.float32)
    std = np.full(param_dim, config.init_std, dtype=np.float32)
    best_params = mean.copy()
    best_shaped_return = float("-inf")
    iteration_summaries = []
    for iteration in range(config.policy_iterations):
        params = rng.normal(mean, std, size=(config.worlds_per_candidate, param_dim)).astype(np.float32)
        shaped_returns, true_returns = rollout_policy_population(
            model=model,
            data=data,
            mjm=mjm,
            params=params,
            reward_expression=reward_expression,
            episode_steps=config.episode_steps,
            dt=dt,
            device=config.device,
            wp=wp,
            mjw=mjw,
        )
        elite_count = max(1, int(config.worlds_per_candidate * config.elite_frac))
        elite_indices = np.argpartition(shaped_returns, -elite_count)[-elite_count:]
        elite_params = params[elite_indices]
        mean = elite_params.mean(axis=0).astype(np.float32)
        std = np.maximum(elite_params.std(axis=0).astype(np.float32), config.min_std)
        best_index = int(np.argmax(shaped_returns))
        if float(shaped_returns[best_index]) > best_shaped_return:
            best_shaped_return = float(shaped_returns[best_index])
            best_params = params[best_index].copy()
        iteration_summaries.append(
            {
                "iteration": iteration,
                "mean_shaped_return": float(np.mean(shaped_returns)),
                "best_shaped_return": float(np.max(shaped_returns)),
                "best_true_return_in_population": float(true_returns[best_index]),
            }
        )
    return best_params, best_shaped_return, iteration_summaries


def train_ppo_policy(
    *,
    model: Any,
    data: Any,
    mjm: Any,
    obs_dim: int,
    action_dim: int,
    reward_expression: "VectorizedRewardExpression",
    config: MjwarpEvaluatorConfig,
    dt: float,
    wp: Any,
    mjw: Any,
) -> tuple[Any, float, list[dict[str, float]]]:
    import torch

    torch_device = torch.device("cuda" if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    policy = AntActorCritic(obs_dim, action_dim).to(torch_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo_learning_rate)
    iteration_summaries = []
    best_shaped_return = float("-inf")

    mjw.reset_data(model, data)
    wp.synchronize()
    dones = np.zeros(config.worlds_per_candidate, dtype=bool)
    episode_shaped = np.zeros(config.worlds_per_candidate, dtype=np.float32)
    episode_true = np.zeros(config.worlds_per_candidate, dtype=np.float32)

    chunks_per_iteration = max(1, int(np.ceil(config.episode_steps / config.ppo_horizon)))
    update_index = 0
    for iteration in range(config.policy_iterations):
        for chunk in range(chunks_per_iteration):
            rollout_steps = min(config.ppo_horizon, config.episode_steps - chunk * config.ppo_horizon)
            batch_obs = []
            batch_actions = []
            batch_logprobs = []
            batch_values = []
            batch_rewards = []
            batch_dones = []
            for _step in range(rollout_steps):
                qpos_before = np.asarray(data.qpos.numpy(), dtype=np.float32)
                qvel_before = np.asarray(data.qvel.numpy(), dtype=np.float32)
                obs_np = ant_policy_obs(qpos_before, qvel_before)
                obs = torch.as_tensor(obs_np, dtype=torch.float32, device=torch_device)
                with torch.no_grad():
                    action_t, logprob_t, _, value_t = policy.act(obs)
                action = action_t.detach().cpu().numpy().astype(np.float32)
                wp.copy(data.ctrl, wp.array(action, dtype=wp.float32, device=config.device))
                mjw.step(model, data)
                wp.synchronize()

                qpos_after = np.asarray(data.qpos.numpy(), dtype=np.float32)
                qvel_after = np.asarray(data.qvel.numpy(), dtype=np.float32)
                shaped_reward, true_reward, next_dones = ant_rewards(
                    qpos_before=qpos_before,
                    qpos_after=qpos_after,
                    qvel_after=qvel_after,
                    action=action,
                    reward_expression=reward_expression,
                    dt=dt,
                )
                active = ~dones
                shaped_reward = np.where(active, shaped_reward, 0.0).astype(np.float32)
                true_reward = np.where(active, true_reward, 0.0).astype(np.float32)
                episode_shaped += shaped_reward
                episode_true += true_reward
                dones |= next_dones

                batch_obs.append(obs_np)
                batch_actions.append(action)
                batch_logprobs.append(logprob_t.detach().cpu().numpy().astype(np.float32))
                batch_values.append(value_t.detach().cpu().numpy().astype(np.float32))
                batch_rewards.append(shaped_reward)
                batch_dones.append(dones.astype(np.float32))

            qpos_before = np.asarray(data.qpos.numpy(), dtype=np.float32)
            qvel = np.asarray(data.qvel.numpy(), dtype=np.float32)
            next_obs = torch.as_tensor(ant_policy_obs(qpos_before, qvel), dtype=torch.float32, device=torch_device)
            with torch.no_grad():
                next_value = policy.value(next_obs).detach().cpu().numpy().astype(np.float32)

            rewards = np.asarray(batch_rewards, dtype=np.float32)
            values = np.asarray(batch_values, dtype=np.float32)
            done_flags = np.asarray(batch_dones, dtype=np.float32)
            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                dones=done_flags,
                next_value=next_value,
                gamma=config.ppo_gamma,
                gae_lambda=config.ppo_gae_lambda,
            )
            flat_obs = torch.as_tensor(np.asarray(batch_obs, dtype=np.float32).reshape(-1, obs_dim), device=torch_device)
            flat_actions = torch.as_tensor(np.asarray(batch_actions, dtype=np.float32).reshape(-1, action_dim), device=torch_device)
            flat_old_logprobs = torch.as_tensor(np.asarray(batch_logprobs, dtype=np.float32).reshape(-1), device=torch_device)
            flat_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=torch_device)
            flat_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=torch_device)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / flat_advantages.std().clamp_min(1e-6)

            ppo_update(
                policy=policy,
                optimizer=optimizer,
                obs=flat_obs,
                actions=flat_actions,
                old_logprobs=flat_old_logprobs,
                advantages=flat_advantages,
                returns=flat_returns,
                config=config,
            )
            mean_shaped = float(np.mean(episode_shaped))
            max_shaped = float(np.max(episode_shaped))
            best_shaped_return = max(best_shaped_return, max_shaped)
            iteration_summaries.append(
                {
                    "iteration": iteration,
                    "ppo_update": update_index,
                    "mean_shaped_return": mean_shaped,
                    "best_shaped_return": max_shaped,
                    "mean_true_return_in_population": float(np.mean(episode_true)),
                }
            )
            update_index += 1
            if dones.all():
                episode_shaped.fill(0.0)
                episode_true.fill(0.0)
                dones.fill(False)
                mjw.reset_data(model, data)
                wp.synchronize()

    policy.eval()
    return policy, best_shaped_return, iteration_summaries


def ant_rewards(
    *,
    qpos_before: np.ndarray,
    qpos_after: np.ndarray,
    qvel_after: np.ndarray,
    action: np.ndarray,
    reward_expression: "VectorizedRewardExpression",
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_velocity = (qpos_after[:, 0] - qpos_before[:, 0]) / dt
    y_velocity = (qpos_after[:, 1] - qpos_before[:, 1]) / dt
    torso_z = qpos_after[:, 2]
    healthy = np.isfinite(qpos_after).all(axis=1) & np.isfinite(qvel_after).all(axis=1)
    healthy &= (torso_z >= 0.2) & (torso_z <= 1.0)
    action_l2 = np.square(action).sum(axis=1)
    forward_reward = x_velocity
    control_cost = 0.5 * 1.0e-2 * action_l2
    survive_reward = np.where(healthy, 1.0, 0.0)
    original_reward = forward_reward + survive_reward - control_cost
    context = {
        "x_velocity": x_velocity,
        "y_velocity": y_velocity,
        "forward_reward": forward_reward,
        "control_cost": control_cost,
        "survive_reward": survive_reward,
        "torso_z": torso_z,
        "action_l2": action_l2,
        "original_reward": original_reward,
        "healthy": healthy,
        "terminated": ~healthy,
        "truncated": np.zeros(len(action), dtype=bool),
    }
    shaped_reward = reward_expression(context)
    return shaped_reward.astype(np.float32), original_reward.astype(np.float32), (~healthy)


def compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[step]
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(
    *,
    policy: Any,
    optimizer: Any,
    obs: Any,
    actions: Any,
    old_logprobs: Any,
    advantages: Any,
    returns: Any,
    config: MjwarpEvaluatorConfig,
) -> None:
    import torch

    batch_size = obs.shape[0]
    minibatch_size = min(config.ppo_minibatch_size, batch_size)
    for _epoch in range(config.ppo_epochs):
        permutation = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            new_logprob, entropy, value = policy.evaluate_actions(obs[idx], actions[idx])
            logratio = new_logprob - old_logprobs[idx]
            ratio = logratio.exp()
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip) * advantages[idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[idx] - value).square().mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + config.ppo_value_coef * value_loss - config.ppo_entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.ppo_max_grad_norm)
            optimizer.step()


def ant_policy_obs(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    return np.concatenate([qpos[:, 2:], qvel], axis=1).astype(np.float32)


def evaluate_policy_in_gym(
    *,
    task: str,
    policy: tuple[str, Any],
    obs_dim: int,
    action_dim: int,
    eval_episodes: int,
    seed: int,
) -> list[float]:
    policy_type, policy_data = policy
    if policy_type == "linear":
        weights = policy_data[: obs_dim * action_dim].reshape(obs_dim, action_dim)
        biases = policy_data[obs_dim * action_dim :]
    else:
        import torch

        torch_policy = policy_data
        torch_device = next(torch_policy.parameters()).device
    returns = []
    for episode in range(eval_episodes):
        env = gym.make(task)
        try:
            obs, _info = env.reset(seed=seed + episode)
            total_reward = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                policy_obs = np.asarray(obs[:obs_dim], dtype=np.float32)
                if policy_type == "linear":
                    action = np.tanh(policy_obs @ weights + biases).astype(np.float32)
                else:
                    obs_t = torch.as_tensor(policy_obs[None, :], dtype=torch.float32, device=torch_device)
                    with torch.no_grad():
                        action = torch_policy.mean_action(obs_t).cpu().numpy()[0].astype(np.float32)
                obs, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
        finally:
            env.close()
        returns.append(total_reward)
    return returns


class AntActorCritic:
    def __new__(cls, obs_dim: int, action_dim: int):
        import torch

        class _AntActorCritic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers = []
                last_dim = obs_dim
                for hidden_dim in (256, 128, 64):
                    layers.append(torch.nn.Linear(last_dim, hidden_dim))
                    layers.append(torch.nn.ELU())
                    last_dim = hidden_dim
                self.backbone = torch.nn.Sequential(*layers)
                self.actor_mean = torch.nn.Linear(last_dim, action_dim)
                self.critic = torch.nn.Linear(last_dim, 1)
                self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

            def forward(self, obs: Any) -> tuple[Any, Any]:
                features = self.backbone(obs)
                return self.actor_mean(features), self.critic(features).squeeze(-1)

            def distribution(self, obs: Any) -> Any:
                mean, _value = self.forward(obs)
                std = self.log_std.exp().expand_as(mean)
                return torch.distributions.Normal(mean, std)

            def act(self, obs: Any) -> tuple[Any, Any, Any, Any]:
                dist = self.distribution(obs)
                raw_action = dist.rsample()
                action = torch.tanh(raw_action)
                logprob = dist.log_prob(raw_action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                value = self.value(obs)
                return action, logprob, entropy, value

            def evaluate_actions(self, obs: Any, action: Any) -> tuple[Any, Any, Any]:
                clipped = action.clamp(-0.999, 0.999)
                raw_action = torch.atanh(clipped)
                dist = self.distribution(obs)
                logprob = dist.log_prob(raw_action).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                value = self.value(obs)
                return logprob, entropy, value

            def value(self, obs: Any) -> Any:
                _mean, value = self.forward(obs)
                return value

            def mean_action(self, obs: Any) -> Any:
                mean, _value = self.forward(obs)
                return torch.tanh(mean)

        return _AntActorCritic()


class VectorizedRewardExpression:
    def __init__(self, expression: str, allowed_names: set[str] | frozenset[str]) -> None:
        RewardExpression(expression, allowed_names)
        parsed = ast.parse(expression, mode="eval")
        parsed = NumpyWhereTransformer().visit(parsed)
        ast.fix_missing_locations(parsed)
        self._code = compile(parsed, "<vectorized-reward-expression>", "eval")
        self._scalar = RewardExpression(expression, allowed_names)

    def __call__(self, values: dict[str, np.ndarray]) -> np.ndarray:
        try:
            result = eval(self._code, {"__builtins__": {}, **NUMPY_FUNCS}, values)
            return np.clip(np.where(np.isfinite(result), result, -100.0), -100.0, 100.0).astype(np.float32)
        except Exception:
            worlds = len(next(iter(values.values())))
            rewards = np.empty(worlds, dtype=np.float32)
            for index in range(worlds):
                row = {key: value[index].item() for key, value in values.items()}
                rewards[index] = self._scalar(row)
            return rewards


class NumpyWhereTransformer(ast.NodeTransformer):
    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        return ast.Call(
            func=ast.Name(id="where", ctx=ast.Load()),
            args=[self.visit(node.test), self.visit(node.body), self.visit(node.orelse)],
            keywords=[],
        )


NUMPY_FUNCS = {
    **ALLOWED_FUNCS,
    "abs": np.abs,
    "min": np.minimum,
    "max": np.maximum,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
    "exp": np.exp,
    "where": np.where,
}
