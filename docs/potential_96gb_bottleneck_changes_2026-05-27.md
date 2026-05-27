# Potential 96 GB GPU Bottleneck Changes

**Timestamp:** 2026-05-27T10:31:10-04:00  
**Scope:** Remaining changes recommended after inspecting the serious-run
`16` candidate, `4096` worlds-per-candidate Ant configuration.

## Purpose

The experiment uses EUREKA reward-program search as the data-generation loop
for RLVR updates to a code model. A large GPU should improve candidate
evaluation throughput without changing the meaning of a candidate score:
each generated reward program must train its own Ant policy, and the resulting
policy must receive a comparable true-environment-return score.

The serious evaluation batch represents:

```text
16 candidates * 4096 worlds = 65,536 active Ant worlds
4096 * 500 * 96 = 196,608,000 control transitions per candidate
3,145,728,000 control transitions per EUREKA generation
15,728,640,000 physics substeps per generation at frame_skip=5
```

Memory capacity is unlikely to be the limiting factor. A local probe of the
full `16 * 4096` simulator shape on an 8 GB RTX 2070 reached:

| Probe | Peak GPU Memory | Throughput |
| --- | ---: | ---: |
| Horizon `32`, rollout only | `2751 MiB` | `88,656` control transitions/s |
| Horizon `32`, one PPO epoch | `4205 MiB` | `88,286` control transitions/s |

These probes are not performance forecasts for a 96 GB GPU. They show that
the serious batch fits comfortably below 96 GB and that correctness and
throughput behavior, rather than VRAM capacity, should drive the next work.

## Priority 1: Correct Episode Reset Semantics

### Problem

Training tracks a done mask for each Ant world. Once a world terminates, its
reward contributions are masked, but its simulator state is not reset and
reused immediately. Training resets the full simulator population only at a
policy-iteration boundary.

With `65,536` worlds, early failures can leave many worlds inactive for the
remaining control steps in an iteration. Reward programs that cause earlier
falls may therefore receive fewer usable training transitions, while still
consuming the same simulator workload. This can weaken policy learning and
distort candidate comparisons.

### Recommended Change

Implement per-world episode turnover:

1. Accumulate completed-episode shaped and true returns before reset.
2. Reset only terminated or truncated worlds to paired initial states.
3. Clear done state for those worlds so they immediately continue generating
   PPO transitions.
4. Preserve identical reset-seed sequencing across candidates within a
   generation.
5. Record completed-episode counts and effective active-transition counts per
   candidate.

### Implementation Risk

Selective MuJoCo Warp resets may interact with captured physics execution. If
the simulator API cannot reset arbitrary indexed worlds without disrupting the
capture path, implement a measured fallback:

- periodic synchronized reset windows at fixed rollout boundaries; or
- separate active and completed-world buffers with compaction at boundaries.

The accepted design must be validated against short runs where Ant policies
terminate frequently. A speedup is not acceptable if it silently discards
large portions of the training budget.

## Priority 2: Calibrate PPO Budget Against Signal Quality

### Problem

The serious default evaluates each reward candidate with `196,608,000`
control transitions. Across `20` RLVR iterations and `3` EUREKA generations,
the experiment schedules:

```text
188,743,680,000 control transitions
943,718,400,000 physics substeps
```

This budget was selected to approximate a public EUREKA transition count, but
this implementation differs in simulator, PPO code, verification process, and
outer RLVR loop. Matching transition count alone does not prove that `96`
policy iterations is the right point for a clear model-update signal.

### Recommended Change

Run a pre-experiment policy-budget ablation on fixed reward programs:

| Policy Iterations | Purpose |
| ---: | --- |
| `4` | Pipeline and failure-mode smoke test only |
| `24` | Early learning and ranking stability check |
| `48` | Intermediate compute-quality tradeoff |
| `96` | Intended serious default |

For each budget, evaluate the same reward programs under multiple generation
seeds and report:

- verified-return mean and variance;
- rank correlation between shorter and longest budgets;
- separation between top candidates relative to variance;
- wall-clock time and GPU utilization.

Select the smallest budget that preserves candidate ordering and produces a
usable group-relative RLVR signal. This directly reduces full-run cost if
learning saturates before `96` iterations.

## Priority 3: Reduce Generated Reward Execution Dispatch

### Problem

During rollout, each control step dispatches each candidate's structured
reward program through a Python loop. For one serious EUREKA generation this
is approximately:

```text
500 * 96 * 16 = 768,000 candidate reward-program dispatches
```

As physics execution becomes faster on a large GPU, repeated Python dispatch
and many small tensor operations can become a larger fraction of elapsed time.

### Recommended Change

Introduce a compiled reward-program execution layer:

1. Keep the current AST validator and allowed reward-variable contract.
2. Convert validated structured expressions into a compact tensor-operation
   representation.
3. Fuse candidates with compatible expression structure into batched Torch
   execution, or compile a bounded Torch function per candidate once per
   generation.
4. Preserve named component reductions required for EUREKA feedback.

### Validation

Compare the optimized evaluator against the existing expression executor on
random Ant state/action tensors and on short policy-training runs. Component
values, summed shaped reward, termination handling, and candidate ranking must
match within an explicit numerical tolerance.

## Priority 4: Gate the Verified Evaluator With Reference Comparisons

### Problem

The verified return is the RLVR target. MJWarp Ant has since been selected as
the target domain, so Gym `Ant-v5` differences are transfer measurements rather
than correctness failures. Stability within MJWarp remains essential.

### Recommended Change

For optional transfer characterization, create a comparison report for
deterministic trained policies:

1. Train a set of fixed reward programs.
2. Evaluate each resulting policy with identical seeds in the accelerated
   verifier and Gym `Ant-v5`.
3. Report per-episode return differences, rank correlation, maximum absolute
   error, and whether elite selection changes.
4. Optionally enforce a documented transfer threshold when Gym comparability
   is a study objective.

The result should be stored with run artifacts as a transfer diagnostic, while
MJWarp candidate ranking is validated directly across repeated common seeds.

## Priority 5: Manage Model and Simulator Phases Explicitly

### Problem

The pipeline has three expensive phases on the same device:

1. code-model generation of reward programs;
2. Ant policy training and verified evaluation;
3. GRPO/LoRA training of the code model.

The phases are sequential, but model and simulator allocations may remain
resident across phase transitions. On a 96 GB GPU this may fit, yet it can
reduce allocator headroom, increase fragmentation risk, and hide which phase
owns memory.

### Recommended Change

Add phase-level resource telemetry and lifecycle controls:

- log allocated and reserved CUDA memory before and after generation, Ant
  evaluation, and RLVR training;
- explicitly release simulator-only state after candidate evaluation;
- determine by measurement whether retaining or reloading the code model
  between phases gives better wall-clock behavior;
- surface phase elapsed time in `pipeline_state.json`.

Do not introduce concurrent LLM training and Ant policy evaluation until their
independent memory and utilization profiles have been recorded on the target
GPU.

## Priority 6: Bound Checkpoint and Result Growth

### Problem

The previous three-iteration test run occupied approximately `6.6 GiB`, with
adapter artifacts accounting for about `6.49 GiB`. A twenty-iteration run can
therefore consume tens of gigabytes before additional metrics or failed-run
restarts.

The serious policy schedule also emits many update-level policy summaries. At
the configured `96` iterations and PPO horizon `32`, there are `1536` summary
points per candidate. Retaining full per-update diagnostics can add substantial
JSON output and rewrite overhead across all RLVR iterations.

### Recommended Change

- retain full adapter checkpoints only at configured milestones and the latest
  resumable point;
- store compact per-generation summaries in ordinary JSON results;
- place detailed diagnostic series in append-only compressed artifacts;
- expose retention settings in the bootstrap script;
- estimate required disk capacity during preflight alongside GPU checks.

Auditability must remain intact: verified rewards, generated completions,
candidate lineage, failures, and resume-critical state must not be pruned.

## Required Benchmark Before Serious Training

Run one full EUREKA generation on the target 96 GB GPU using the intended
population and policy settings. Record:

| Measurement | Reason |
| --- | --- |
| Wall-clock time per generation | Predict full experiment duration |
| Peak and reserved GPU memory by phase | Establish capacity margin |
| GPU utilization and power during policy training | Identify underuse |
| Verified evaluator agreement report | Validate RLVR reward semantics |
| Effective active transitions per candidate | Detect reset-related loss |
| Artifact growth after generation completion | Plan storage and resume policy |

Only after this benchmark should the full `20`-iteration RLVR run be treated
as a production experiment rather than a costly calibration pass.
