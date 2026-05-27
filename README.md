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

Design rationale is documented in [docs/design_decisions.md](docs/design_decisions.md).
All run flags are listed in [docs/command_line_reference.md](docs/command_line_reference.md).
The GPU-resident Ant PPO implementation and remaining validation work are tracked in
[docs/gpu_rollout_optimization_plan.md](docs/gpu_rollout_optimization_plan.md).
The paired-seed and policy-budget settings used for stable RLVR rewards are
documented in [docs/eureka_signal_stability.md](docs/eureka_signal_stability.md).
The generation-level GPU candidate batching change is documented in
[docs/candidate_batched_gpu_evaluation_change.md](docs/candidate_batched_gpu_evaluation_change.md).
The episode-reset, verification-audit, calibration, and retention controls are
documented in
[docs/gpu_pipeline_safeguards_2026-05-27.md](docs/gpu_pipeline_safeguards_2026-05-27.md).
The search-state, resume, and canonical RLVR publication refactor is documented
in [docs/search_refactor_2026-05-27.md](docs/search_refactor_2026-05-27.md).

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
and Ant-world batch size, with 3 EUREKA generations and 4 elites per RLVR
iteration, but shortens inner PPO training for validation only.

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --run-root runs/smoke_96gb_mjwarp_rlvr \
  --iterations 3 \
  --mjwarp-policy-iterations 4
```

## Serious Run

Use this for the intended 96 GB GPU experiment: 20 RLVR iterations, 16 reward
candidates per EUREKA generation, 3 EUREKA generations per RLVR iteration, 4
ranked elites in each refinement prompt, 4096 Ant worlds per candidate, and one
PPO actor-critic network per reward candidate. Each reward candidate receives
`196,608,000` Ant control transitions (`4096 * 500 * 96`), and candidates in
the same EUREKA generation use a common policy/evaluation seed for paired
ranking. The default MJWarp evaluator is
`ppo`; the older lightweight evaluator remains available with
`--mjwarp-evaluator search`.
PPO rollouts and verified reward evaluation use the GPU-resident MuJoCo Warp
Ant environment by default. This is the target domain for the experiment:
RLVR trains the code model to produce reward programs that yield strong Ant
policies in MJWarp. Gym `Ant-v5` evaluation is an optional transfer diagnostic,
not the objective being optimized.
The GPU PPO path batches the `16` candidate policies into one simulator job
(`65536` total worlds) while retaining `4096` worlds and one independent policy
per candidate. Completed Ant worlds reset immediately in place while preserving
the PPO transition terminal flag; `--mjwarp-training-episode-horizon 1000`
keeps training episodes aligned with the Ant time limit independently of the
`500`-step PPO collection iteration. MuJoCo physics substeps use CUDA graph
replay by default.

To measure transfer to the Gym reference implementation, run an optional audit:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --run-root runs/audit_96gb_mjwarp_rlvr \
  --iterations 1 \
  --mjwarp-policy-iterations 4 \
  --mjwarp-verified-audit-gym
```

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --iterations 20
```

The serious run defaults can be changed with `--generations`, `--eureka-elites`,
`--population`, `--worlds-per-candidate`, and `--mjwarp-policy-iterations`.
Use `--no-mjwarp-candidate-batching` or `--no-mjwarp-cuda-graph` for
sequential/capture-disabled ablations.

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

To exclude invalid code and failed evaluations from model updates for an
ablation run:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --iterations 20 --no-negative-rlvr-samples
```

For multi-epoch trainer runs with bounded resume storage, retain only the most
recent complete RLVR checkpoint:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --iterations 20 --checkpoint-retention latest
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

The process exits cleanly after the current candidate batch or trainer epoch.
With the default GPU batching, the batch is one EUREKA generation; pass
`--no-mjwarp-candidate-batching` for candidate-by-candidate pause boundaries.

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
