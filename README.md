# Training Models to Train Models

This is a small EUREKA-inspired RLVR experiment. The goal is to train a coding
model to generate better reward functions for reinforcement learning.

The loop is:

1. Sample reward-code candidates from a coding model.
2. Use each reward candidate to train Ant policies in MuJoCo Warp.
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

## Small GPU

Use this for an 8 GB GPU smoke run. It verifies the full pipeline at reduced
scale.

```bash
ALLOW_SMALL_GPU=1 \
RUN_ROOT=runs/smoke_8gb_mjwarp_rlvr \
ITERATIONS=1 \
POPULATION=2 \
WORLDS_PER_CANDIDATE=128 \
MJWARP_EPISODE_STEPS=25 \
MJWARP_POLICY_ITERATIONS=1 \
EVAL_EPISODES=1 \
MAX_NEW_TOKENS=96 \
RLVR_EPOCHS=1 \
./scripts/run_full_mjwarp_rlvr_96gb.sh
```

## 96 GB GPU

Use this for the full intended run. Defaults are 3 RLVR iterations, 16 reward
candidates per iteration, and 4096 Ant worlds per candidate.

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh
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
./scripts/run_full_mjwarp_rlvr_96gb.sh
```

For the 8 GB smoke run, use its run root:

```bash
touch runs/smoke_8gb_mjwarp_rlvr/PAUSE
rm runs/smoke_8gb_mjwarp_rlvr/PAUSE
```
