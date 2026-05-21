# Training Models to Train Models

This is a small EUREKA-inspired RLVR experiment. The goal is to train a coding
model to generate better reward functions for reinforcement learning.

The loop is:

1. Sample reward-code candidates from a coding model.
2. Use each reward candidate to train an Ant PPO policy in MuJoCo Warp.
3. Evaluate the resulting policies with the true `Ant-v5` environment return.
4. Store prompt, completion tokens, old logprobs, and verified return as RLVR data.
5. Train a LoRA adapter with GRPO, then use that adapter to sample the next round.

The default full pipeline is iterative: each iteration evaluates a batch of
reward candidates, trains an adapter from verified EUREKA performance, and uses
the new adapter in the next iteration.

## Setup

From a fresh clone, the run script handles the environment setup automatically:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh
```

It creates `.venv`, installs PyTorch, installs this repo with MuJoCo Warp
support, verifies CUDA/Warp availability, runs a small Ant smoke test, and then
starts the training pipeline.

If you want to install manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch
python -m pip install -e ".[mjwarp]"
```

## Smoke Test

Use this for a 3-iteration smoke test on a 96 GB GPU. It uses the full candidate
and Ant-world batch size, but fewer RLVR iterations than the serious run.

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --run-root runs/smoke_96gb_mjwarp_rlvr \
  --iterations 3
```

## Serious Run

Use this for the intended 96 GB GPU experiment: 20 RLVR iterations, 16 reward
candidates per iteration, 4096 Ant worlds per candidate, and one PPO
actor-critic network per reward candidate. The default MJWarp evaluator is
`ppo`; the older lightweight evaluator remains available with
`--mjwarp-evaluator search`. The PPO policy uses a shared MLP `[256, 128, 64]`
with ELU activations, rollout horizon `32`, minibatch size `16384`, 4 PPO epochs,
learning rate `3e-4`, GAE `0.95`, and clip range `0.2`.

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --iterations 20
```

Outputs are written under:

```text
runs/deepseek_lite_ant_mjwarp_rlvr/
```

## Pause And Resume

Pause a running 96 GB pipeline:

```bash
touch runs/deepseek_lite_ant_mjwarp_rlvr/PAUSE
```

The process exits cleanly after the current reward candidate or trainer epoch.

Resume:

```bash
rm runs/deepseek_lite_ant_mjwarp_rlvr/PAUSE
./scripts/run_full_mjwarp_rlvr_96gb.sh --iterations 20
```

For the smoke test, use its run root:

```bash
touch runs/smoke_96gb_mjwarp_rlvr/PAUSE
rm runs/smoke_96gb_mjwarp_rlvr/PAUSE
```
