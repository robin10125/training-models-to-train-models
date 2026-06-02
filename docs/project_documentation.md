# Project Documentation

Consolidated: 2026-05-28 (America/Toronto)

This file condenses the previous project documentation into one dated
reference. Sections marked "Undated" came from docs that did not carry a
timestamp. Sections marked "2026-05-27" came from dated filenames or explicit
timestamps in the source docs.

The experiment invariants that future implementation work must preserve are
defined in [experiment_constitution.md](experiment_constitution.md).

## Source Index

| Date | Former document | Condensed topic |
| --- | --- | --- |
| Undated | `experimental_structure_report.txt` | Research motivation and experimental structure |
| Undated | `design_decisions.md` | EUREKA/RLVR semantics and reward design |
| Undated | `command_line_reference.md` | CLI commands and flags |
| Undated | `eureka_signal_stability.md` | Stable candidate ranking settings |
| Undated | `gpu_rollout_optimization_plan.md` | GPU-resident PPO architecture |
| Undated | `candidate_batched_gpu_evaluation_change.md` | Generation-level candidate batching |
| 2026-05-27 | `gpu_pipeline_safeguards_2026-05-27.md` | Reset, audit, calibration, telemetry, retention |
| 2026-05-27 | `search_refactor_2026-05-27.md` | Search-state refactor and correctness fixes |
| 2026-05-27 | `potential_96gb_bottleneck_changes_2026-05-27.md` | Target-GPU bottleneck recommendations |
| 2026-05-27 | `rtx2070_performance_benchmark_2026-05-27.md` | Local RTX 2070 benchmark |
| 2026-05-27 | `base_policy_warm_start_plan_2026-05-27.md` | PPO warm-start plan |

## Undated: Experimental Structure

This project is an EUREKA-inspired RLVR experiment. The goal is to train a code
model to generate better reinforcement-learning reward functions. Ant
locomotion is used as a compact, measurable task for studying reward-code
generation under a verifiable downstream signal.

The core hypothesis is that reward-code generation can be treated as a
reinforcement-learning problem for the language model:

1. The model samples a reward program.
2. The reward program trains an Ant PPO policy in MuJoCo Warp.
3. The trained policy is evaluated with the original MJWarp Ant return.
4. That verified return is assigned as the RLVR reward for the model completion.
5. GRPO/LoRA updates the code model, and the updated model samples the next
   iteration's reward programs.

The experiment contains two coupled loops:

- Inner loop: EUREKA-style evolutionary reward search.
- Outer loop: RLVR training of the code model from verified search outcomes.

This is not an exact reproduction of public EUREKA. It combines EUREKA-style
reward discovery with MJWarp Ant policy training and an outer model-update
loop. The central extension is training the code model from EUREKA performance.

## Undated: EUREKA And RLVR Design Decisions

Each RLVR iteration contains multiple EUREKA generations. In each generation:

1. The code model samples structured reward programs.
2. Each executable reward program trains one Ant PPO policy over parallel
   MJWarp worlds.
3. Policies are evaluated with the original Ant reward in MJWarp.
4. Candidates are ranked by verified return.
5. Elites, lower-ranked examples, source context, reward components, verified
   scores, and policy diagnostics are fed into the next generation prompt.

Prompts include reward-relevant Ant source excerpts, the task adapter, and the
local MJWarp reward/evaluation code. The model is asked for named reward
components such as:

```python
{"forward": "x_velocity", "healthy": "survive_reward", "control": "-0.01 * action_l2"}
```

The evaluator validates each component, sums the components for PPO training,
and records component-level statistics for reflection. Scalar reward
expressions are still accepted and treated as a single `total` component.

`verified_reward` means the target-domain MJWarp Ant return obtained by the
trained policy. It is not the shaped reward. For successful candidates:

```text
rlvr_reward = verified MJWarp Ant return
```

For invalid generated programs or failed evaluations, negative RLVR samples are
enabled by default. Their penalty is:

```text
penalty = worst_successful_verified_reward - negative_rlvr_margin
```

If no candidate succeeds in a generation, the penalty is
`-negative_rlvr_margin`.

Negative samples retain prompt text, raw sampled completion, token IDs, old log
probabilities, error text, lineage, and assigned penalty. They can be disabled
with:

```bash
--no-negative-rlvr-samples
```

GRPO compares completions generated from the same prompt and model checkpoint.
Retry prompts after validator feedback form separate prompt groups and only
contribute to GRPO when the group has at least two trainable samples.

Checkpoint and audit records preserve source context, prompt context, raw
completion, reward components, lineage, elite archive, rank, verified return,
RLVR reward, failures, and PPO/MJWarp diagnostics.

## Undated: Command Line Reference

Full pipeline:

```bash
python -m eureka_lite.pipeline [options]
```

Main experiment flags:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model-id` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | HF code model. |
| `--run-root` | `runs/deepseek_lite_ant_mjwarp_rlvr` | Output directory. |
| `--iterations` | `3` | Outer RLVR sample/evaluate/train iterations. |
| `--population` | `16` | Reward programs per EUREKA generation. |
| `--generations` | `3` | EUREKA generations per RLVR iteration. |
| `--eureka-elites` | `4` | Elite programs retained for refinement. |
| `--eval-episodes` | `5` | Verified episodes per candidate. |
| `--seed` | `7` | Base random seed. |
| `--device` | `cuda` | `auto`, `cpu`, or `cuda`. |

MJWarp/Ant flags:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--worlds-per-candidate` | `4096` | Parallel Ant worlds per reward program. |
| `--mjwarp-evaluator` | `ppo` | `ppo` or legacy `search`. |
| `--mjwarp-episode-steps` | `500` | PPO control steps per policy iteration. |
| `--mjwarp-training-episode-horizon` | `1000` | Training episode timeout before reset. |
| `--mjwarp-policy-iterations` | `96` in Python CLI, `32` in 96 GB script | Candidate PPO iterations. |
| `--mjwarp-ppo-horizon` | `32` | PPO rollout horizon before update. |
| `--mjwarp-ppo-epochs` | `4` | PPO optimization epochs. |
| `--mjwarp-ppo-minibatch-size` | `16384` | PPO minibatch size. |
| `--mjwarp-ppo-learning-rate` | `3e-4` | PPO learning rate. |
| `--mjwarp-ppo-init-mode` | `base` | Shared pretrained `base` or cold-start `scratch`. |
| `--mjwarp-base-policy-checkpoint` | `checkpoints/base_ant_mjwarp_policy.pt` | Base checkpoint for `base` mode. |
| `--mjwarp-rollout-mode` | `gpu` | `gpu` or `host`. |
| `--mjwarp-verified-evaluator` | `mjwarp` | Target `mjwarp` or transfer `gym`. |
| `--mjwarp-verification-steps` | `1000` | Verified rollout horizon. |
| `--mjwarp-verified-audit-gym` | off | Add Gym transfer diagnostics. |
| `--mjwarp-verified-audit-max-abs-diff` | none | Fail above this Gym/MJWarp diff. |
| `--mjwarp-reward-backend` | `eager` | `eager` or optional `compiled`. |
| `--no-mjwarp-candidate-batching` | off | Evaluate candidates sequentially. |
| `--no-mjwarp-cuda-graph` | off | Disable physics CUDA graph replay. |

Generation and model flags:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--max-new-tokens` | `256` | Completion length. |
| `--temperature` | `0.7` | Sampling temperature. |
| `--top-p` | `0.95` | Nucleus sampling. |
| `--no-4bit` | off | Disable 4-bit model loading. |
| `--no-negative-rlvr-samples` | off | Exclude invalid/failed completions. |
| `--negative-rlvr-margin` | `1.0` | Negative sample penalty margin. |

RLVR trainer flags:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--trainer-algorithm` | `grpo` | `grpo` or `weighted_sft`. |
| `--trainer-epochs` | `1` | LoRA epochs per iteration. |
| `--trainer-batch-size` | `1` | Trainer batch size. |
| `--trainer-learning-rate` | `5e-5` | LoRA learning rate. |
| `--trainer-max-length` | `8192` | Max token sequence length. |
| `--trainer-max-grad-norm` | `1.0` | Gradient clipping norm. |
| `--trainer-lora-r` | `16` | LoRA rank. |
| `--trainer-lora-alpha` | `32` | LoRA alpha. |
| `--trainer-lora-dropout` | `0.05` | LoRA dropout. |
| `--trainer-clip-epsilon` | `0.2` | GRPO clip epsilon. |
| `--trainer-beta-kl` | `0.01` | GRPO KL coefficient. |
| `--overwrite-collection` | off | Replace collection artifacts. |
| `--force-train` | off | Re-run adapter training. |
| `--checkpoint-retention` | `all` | `all` or `latest`. |

96 GB bootstrap script:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh [options]
```

Extra script-only flags:

| Argument | Meaning |
| --- | --- |
| `--pretrain-base-policy` | Train a base policy if missing. |
| `--force-pretrain-base-policy` | Retrain base policy even if present. |
| `--mjwarp-base-policy-iterations N` | PPO iterations used to pretrain the base policy; default `96`. |
| `--cold-start` | Use scratch PPO initialization and 96 candidate iterations. |
| `--allow-small-gpu` | Bypass 96 GB memory preflight. |
| `--no-smoke-test` | Skip startup MJWarp smoke test. |

Base policy pretraining:

```bash
python -m eureka_lite.pretrain_mjwarp_ant_policy \
  --output checkpoints/base_ant_mjwarp_policy.pt \
  --worlds-per-candidate 4096 \
  --mjwarp-policy-iterations 96
```

PPO budget calibration:

```bash
python -m eureka_lite.calibrate_mjwarp \
  --population 16 \
  --worlds-per-candidate 4096 \
  --budgets 4 24 48 96 \
  --seeds 7 17 27
```

Standalone reward search without outer model updates:

```bash
python -m eureka_lite [options]
```

## Undated: EUREKA Signal Stability

The RLVR update should learn from differences in reward programs, not avoidable
randomness in Ant policy training.

Cold-start reference runs use:

```text
4096 worlds * 500 control steps * 96 policy iterations
  = 196,608,000 Ant control transitions per reward candidate
```

This aligns with public EUREKA's Ant transition count implied by `4096` actors,
PPO horizon `16`, and `3000` RL policy iterations. It is a transition-budget
alignment, not an exact Isaac Gym reproduction.

Within an EUREKA generation, executable candidates receive the same evaluation
seed:

```text
candidate_seed = base_seed + generation * 100
```

Torch RNG is reset before policy construction and stochastic action sampling.
Final verified episodes derive from the same seed. In the batched GPU evaluator,
candidate policy slices are initialized identically and receive common
exploration noise for corresponding worlds; they diverge through
candidate-specific shaped rewards and PPO updates.

Default serious run:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --iterations 20
```

For smoke tests, reduce policy iterations explicitly:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --iterations 3 \
  --mjwarp-policy-iterations 4
```

## Undated: GPU-Resident Ant PPO Optimization

The target architecture keeps MuJoCo state, Torch policy observations/actions,
rewards, masks, rollout buffers, GAE, and PPO updates on CUDA for an entire PPO
horizon. CPU transfers should occur only for compact summaries, checkpoints,
and final ranked results.

Implemented capabilities:

- GPU-resident PPO rollout (`--mjwarp-rollout-mode gpu`, default).
- Torch CUDA generated-reward execution and GAE.
- Torch/Warp shared-memory control transfer.
- Frame-skipped MJWarp stepping matching Ant action cadence.
- Batched MJWarp verified evaluation.
- Generation-level candidate-batched PPO evaluation.
- Best-effort CUDA graph replay for repeated Ant physics substeps.
- Host rollout and Gym transfer-reference paths.

Important constraints:

- Each reward candidate trains an independent PPO actor-critic.
- Generated reward components affect policy training only.
- Verified reward remains original MJWarp Ant return.
- Component-level shaped-reward statistics remain available for EUREKA
  feedback.
- Invalid and failed completions remain usable negative RLVR samples.
- Checkpoint/resume behavior must remain valid.

Remaining measurement work:

- Characterize MJWarp rank stability and optional Gym transfer.
- Profile world-count scaling on the target 96 GB GPU.
- Benchmark a full `16 * 4096` candidate batch on the target GPU before
  treating a 20-iteration run as production.

## Undated: Candidate-Batched GPU Evaluation

The default `mjwarp` + `ppo` + `gpu` path evaluates all executable reward
candidates in one generation-level GPU job. Serious defaults create:

```text
16 candidates * 4096 worlds per candidate = 65536 total simulated worlds
```

The flattened MJWarp state is reshaped as `[candidate, world, state]`.
Each candidate owns an independent actor-critic parameter slice and independent
optimizer gradients. A candidate's generated reward program is applied only to
that candidate's worlds. Verified returns and RLVR records remain per-candidate.

This is candidate batching, not pooling. It does not increase the per-candidate
world count and does not change the training budget.

Candidates share initial policy values and common exploratory action noise under
the generation seed, which makes within-generation ranking more attributable to
reward-code differences.

Controls:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --no-mjwarp-candidate-batching
./scripts/run_full_mjwarp_rlvr_96gb.sh --no-mjwarp-cuda-graph
```

Pause boundaries are one generation-level candidate batch when batching is
enabled, or one candidate when batching is disabled.

## 2026-05-27: GPU Pipeline Safeguards

GPU PPO now records terminal transitions and immediately resets finished worlds
through MJWarp masked reset. This prevents terminated Ant worlds from remaining
inactive until a full policy-iteration boundary.

Training rollout budget and episode lifetime are separate:

- `--mjwarp-episode-steps` controls transitions collected per PPO policy
  iteration.
- `--mjwarp-training-episode-horizon` controls training episode timeout and
  defaults to `1000`.

MJWarp is the production verified-return domain. Gym `Ant-v5` is optional
transfer metadata:

```bash
--mjwarp-verified-audit-gym
--mjwarp-verified-audit-max-abs-diff X
```

`python -m eureka_lite.calibrate_mjwarp` evaluates fixed candidate populations
across PPO budgets and common seeds. Choose budgets by stable rank behavior and
elite agreement, not raw throughput alone.

`--mjwarp-reward-backend eager` is the reference path. `compiled` is optional
and falls back to eager if compilation fails.

Pipeline summaries include phase elapsed time and CUDA memory telemetry.
`--checkpoint-retention all|latest` controls RLVR epoch checkpoint retention.

Recommended serious-run procedure:

1. Run a short MJWarp smoke run.
2. Run policy-budget calibration over common seeds.
3. Choose the smallest budget with stable rank behavior.
4. Start the long RLVR run with MJWarp verification.
5. Enable compiled reward execution only after representative benchmarking.

## 2026-05-27: Search Refactor And Correctness

`search.py` is now orchestration. Focused modules own specific concerns:

- `search_types.py`: result/config schemas and `MjwarpOptions`.
- `search_state.py`: resumable generation state.
- `search_artifacts.py`: checkpoint compatibility, result publication, RLVR
  serialization, and event artifacts.
- `search_feedback.py`: penalties, ranking, elite context, and feedback.

Correctness fixes:

- Resume now persists explicit generation phase (`needs_population` or
  `evaluating`) so HF resume does not regenerate stale generation-zero
  candidates at later boundaries.
- Batched GPU results are committed in one checkpoint transaction per group.
- `rlvr_records.jsonl` is rewritten only after generation finalization assigns
  penalties, separates invalid completions, ranks candidates, and annotates
  elites.
- Invalid completions can become negative RLVR samples but are never EUREKA
  elites or mutation parents.
- GPU summaries now include device-side best internal target-environment return
  for evolution feedback.

Flat public `mjwarp_*` fields remain for CLI and checkpoint compatibility, but
translation is centralized through `MjwarpOptions`.

## 2026-05-27: Potential 96 GB Bottleneck Changes

The serious evaluation shape is:

```text
16 candidates * 4096 worlds = 65,536 active Ant worlds
4096 * 500 * 96 = 196,608,000 control transitions per candidate
3,145,728,000 control transitions per EUREKA generation
15,728,640,000 physics substeps per generation at frame_skip=5
```

Memory capacity is unlikely to be the only bottleneck. The accepted speedups
must preserve candidate-score meaning: each reward program trains its own Ant
policy and receives a comparable original-return score.

Recommended priorities:

1. Correct episode reset semantics with per-world turnover.
2. Calibrate PPO budget against candidate-rank stability.
3. Reduce generated-reward dispatch if target-GPU profiling shows many small
   reward kernels dominate.
4. Treat Gym as optional transfer comparison, not MJWarp correctness.
5. Manage model/simulator phases with explicit memory telemetry.
6. Bound checkpoint and result growth.

Before serious training on the target 96 GB GPU, run one full EUREKA generation
and record wall-clock time, peak/reserved memory by phase, GPU utilization,
verified evaluator diagnostics, effective active transitions, and artifact
growth.

## 2026-05-27: RTX 2070 MJWarp Benchmark

Local benchmark hardware:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2070 |
| VRAM | 8192 MiB |
| Driver | 595.58.03 |
| Compute capability | 7.5 |
| PyTorch | 2.11.0+cu130 |
| Warp | 1.13.0 |

World scaling with `16` candidates, `32` control steps, `1` policy iteration,
`0` PPO epochs, eager rewards, CUDA graph enabled:

| Worlds/Candidate | Total Worlds | Seconds | Control Transitions/s |
| ---: | ---: | ---: | ---: |
| 512 | 8,192 | 3.5365 | 74,125 |
| 1,024 | 16,384 | 5.4467 | 96,258 |
| 2,048 | 32,768 | 9.5959 | 109,273 |
| 4,096 | 65,536 | 18.0555 | 116,150 |

At `16 x 4096`, four PPO epochs added only about `3.8%` wall time over
rollout-only execution. Physics/control/reward rollout dominates on the RTX
2070. CUDA graph replay improved the short test by about `2.2%`.

Candidate batching matters. In a `4 x 1024` test, batched evaluation was about
`2.54x` faster than sequential evaluation.

Compiled reward backend did not outperform eager execution in the benchmark.
Keep `--mjwarp-reward-backend eager` as the production default until reward
execution is represented in a compiler-friendly fused form.

Runtime estimate from the longer full-batch RTX 2070 case:

| Scope | Ant Control Transitions | Estimated Collection Time |
| --- | ---: | ---: |
| One EUREKA generation | 3,145,728,000 | 7.69 hours |
| `20` iterations x `3` generations | 188,743,680,000 | 19.2 days |

These estimates exclude code-model sampling, GRPO updates, checkpoint I/O, and
interruptions. They should not be used as direct 96 GB GPU forecasts.

## 2026-05-27: Base-Policy PPO Warm Starts

Base-policy warm start is a throughput optimization:

1. Pretrain one competent Ant policy with original MJWarp Ant reward.
2. Initialize every candidate policy from that same checkpoint.
3. Use a shorter candidate-specific PPO fine-tuning budget.

This changes the inner question from:

```text
Can this reward train Ant from scratch?
```

to:

```text
Can this reward fine-tune a competent Ant policy into better behavior?
```

Supported initialization modes:

| Mode | Meaning | Use |
| --- | --- | --- |
| `base` | Every candidate starts from the same pretrained original-reward policy. | Default large-scale RLVR mode. |
| `scratch` | Seeded random policy initialization. | Cold-start reference and EUREKA-style checks. |

Planned future mode:

| Mode | Meaning |
| --- | --- |
| `lineage` | Refined candidates inherit parent policy checkpoints. |

Pretrain a base policy:

```bash
python -m eureka_lite.pretrain_mjwarp_ant_policy \
  --output checkpoints/base_ant_mjwarp_policy.pt \
  --worlds-per-candidate 4096 \
  --mjwarp-policy-iterations 96
```

Use default base initialization:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --iterations 20
```

Use cold-start reference initialization:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --iterations 20 \
  --cold-start
```

Candidate fine-tuning budget examples:

| Candidate Iterations | Transition Budget | Budget Speedup vs 96 |
| ---: | ---: | ---: |
| 20 | 40,960,000 | 4.8x |
| 24 | 49,152,000 | 4.0x |
| 32 | 65,536,000 | 3.0x |
| 48 | 98,304,000 | 2.0x |

Validation requirement:

- Compare `base` versus `scratch` rankings across common seeds.
- Report Spearman rank correlation and top-k elite agreement.
- Look for rewards that are good from scratch but hidden by base fine-tuning.
- Look for rewards that merely preserve the base policy without improving
  learning.

Current default:

```text
default large-GPU script mode: base warm start
reference cold-start mode: scratch with 96 candidate iterations
```

Warm starts must not change the scalar RLVR reward. Successful candidates still
receive verified MJWarp Ant return, not shaped reward or improvement over base.

## 2026-05-28: RTX PRO 6000 Blackwell Run Notes

The intended 96 GB target is the NVIDIA RTX PRO 6000 Blackwell. The older
RTX 6000 Ada Generation is a 48 GB card and should use smaller settings or
`--allow-small-gpu`.

Smoke test:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --run-root runs/smoke_96gb_mjwarp_rlvr \
  --iterations 3 \
  --mjwarp-policy-iterations 4
```

Default warm-start run:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --run-root runs/deepseek_lite_ant_mjwarp_rlvr \
  --iterations 20
```

Cold-start reference run:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --run-root runs/deepseek_lite_ant_mjwarp_rlvr_cold \
  --iterations 20 \
  --cold-start
```

Pause:

```bash
touch runs/deepseek_lite_ant_mjwarp_rlvr/PAUSE
```

Resume:

```bash
rm runs/deepseek_lite_ant_mjwarp_rlvr/PAUSE
./scripts/run_full_mjwarp_rlvr_96gb.sh --iterations 20
```
