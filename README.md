# Training Models to Train Models

This is a small EUREKA-inspired RLVR experiment. The goal is to train a coding
model to generate better reward functions for reinforcement learning.

The loop is:

1. Sample reward-code candidates from a coding model.
2. Run a full EUREKA search inside each RLVR iteration: rank candidates, keep
   elites, and feed task context plus evolutionary feedback into the next
   refinement prompt.
3. Use each reward candidate to train an Ant PPO policy in MuJoCo Warp.
4. Evaluate the resulting policies with the original Ant task reward in the
   MJWarp target environment.
5. Store structured reward components, prompt, completion tokens, old logprobs,
   EUREKA lineage, elite context, reflection feedback, and verified return as
   RLVR data.
6. Train a LoRA adapter with GRPO, then use that adapter to sample the next
   round.

The default full pipeline is iterative: each iteration evaluates a batch of
reward candidates, trains an adapter from verified EUREKA performance, and uses
the new adapter in the next iteration. Prompts include Ant task source excerpts
and ask the model for named reward components so the evaluator can report
component-level statistics during reflection. Invalid generated reward code and
failed evaluations are included as penalized RLVR samples by default.

The experiment invariants that must be preserved by future changes are defined
in [docs/experiment_constitution.md](docs/experiment_constitution.md). Design
rationale, command-line flags, GPU execution notes, benchmark summaries, and
dated change records are consolidated in
[docs/project_documentation.md](docs/project_documentation.md).

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

## RTX PRO 6000 Blackwell Smoke Test

Use this first on an NVIDIA RTX PRO 6000 Blackwell 96 GB GPU. It runs the full
16-candidate, 4096-world MJWarp shape, but uses a short PPO budget so you can
verify setup, checkpointing, generation, evaluation, and RLVR training.

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --run-root runs/smoke_96gb_mjwarp_rlvr \
  --iterations 3 \
  --mjwarp-policy-iterations 4
```

## RTX PRO 6000 Blackwell Serious Run

Use this for the default RTX PRO 6000 Blackwell 96 GB experiment. The default
run uses the saved `1500`-gate MJWarp Ant warm-start policy at
`checkpoints/ant_mjwarp_warm_start_1500.pt` and fine-tunes each reward
candidate from that checkpoint.

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh
```

Defaults for this command:

- `16` candidates per EUREKA generation
- `3` EUREKA generations per RLVR iteration
- `4` elites carried into refinement prompts
- `4096` MJWarp Ant worlds per candidate
- one shared early-locomotion Ant warm-start policy with verified mean return
  about `1724`
- `32` PPO fine-tuning iterations per candidate
- conservative MJWarp verified score as the RLVR reward:
  `mean_return - 0.25 * std_return`
- `32` common-seed verification episodes for serious ranking
- GRPO LoRA updates after each collection iteration

Use this for the cold-start reference path. It trains every reward candidate
from seeded random PPO initialization with the full `96`-iteration budget:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --run-root runs/deepseek_lite_ant_mjwarp_rlvr_cold \
  --cold-start
```

To reuse an existing base-policy checkpoint:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --run-root runs/deepseek_lite_ant_mjwarp_rlvr_base \
  --mjwarp-base-policy-checkpoint checkpoints/ant_mjwarp_warm_start_1500.pt \
  --mjwarp-policy-iterations 32
```

The warm-start path is faster, but it changes the inner question from training
Ant from scratch to fine-tuning a competent Ant policy. Use the from-scratch
command for reference runs. The equivalent explicit cold-start flags are:

```bash
--mjwarp-ppo-init-mode scratch --mjwarp-policy-iterations 96
```

To measure transfer to Gym `Ant-v5` as a diagnostic, add:

```bash
--mjwarp-verified-audit-gym
```

To exclude invalid code and failed evaluations from model updates for an
ablation run, add:

```bash
--no-negative-rlvr-samples
```

For bounded trainer checkpoint storage, add:

```bash
--checkpoint-retention latest
```

Calibrate how much Ant policy training is needed before changing the serious
run budget. This evaluates fixed reward candidates and does not update the code
model:

```bash
.venv/bin/python -m eureka_lite.calibrate_mjwarp \
  --population 16 \
  --worlds-per-candidate 4096 \
  --budgets 4 24 48 96 \
  --seeds 7 17 27 \
  --mjwarp-verified-audit-gym
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

With the default GPU batching, the batch is one EUREKA generation; pass
`--no-mjwarp-candidate-batching` for candidate-by-candidate pause boundaries.

Resume:

```bash
rm runs/deepseek_lite_ant_mjwarp_rlvr/PAUSE
./scripts/run_full_mjwarp_rlvr_96gb.sh
```

For the smoke test, use its run root:

```bash
touch runs/smoke_96gb_mjwarp_rlvr/PAUSE
rm runs/smoke_96gb_mjwarp_rlvr/PAUSE
```
