# GPU-Resident Ant PPO Optimization Plan

## Implementation Status

Implemented:

- selectable GPU-resident PPO rollout (`--mjwarp-rollout-mode gpu`, default);
- Torch CUDA generated-reward execution and GAE;
- Torch/Warp shared-memory policy-to-simulator control transfer;
- removal of per-step `numpy()` transfers and explicit synchronization from
  the GPU PPO loop;
- frame-skipped MJWarp control stepping matching Ant action cadence;
- selectable batched MJWarp verified evaluation with fixed-seed Gym reset
  initialization (`--mjwarp-verified-evaluator mjwarp`);
- generation-level candidate-batched PPO evaluation with independent policy
  parameters and `4096` worlds per candidate;
- best-effort CUDA graph replay for repeated Ant physics substeps;
- host rollout and Gym transfer-reference paths.

Still to be completed through measurement rather than assumed:

- MJWarp rank-stability characterization and optional Gym transfer reporting;
- target 96 GB GPU world-count scaling;
- full target-GPU throughput measurement of the `16 * 4096` candidate batch.

## Purpose

The intended experiment evaluates many code-model reward proposals by training
one Ant policy for each proposal in MuJoCo Warp, then assigning verified
MJWarp Ant return to the corresponding model completion. On a 96 GB GPU, the
useful scaling target is not simply fitting more worlds in memory. The Ant PPO
evaluator must keep rollout work on CUDA long enough for the GPU to execute it
efficiently.

This document plans the changes required to make the Ant policy-training phase
GPU-resident, preserve the EUREKA/RLVR experimental semantics, and establish
whether candidate-level concurrency is useful after the evaluator itself is
efficient.

## Current Finding

The existing PPO evaluator is host-orchestrated and strongly limited by
per-timestep synchronization and data transfer.

Local measurements were taken on an RTX 2070 with an AMD Ryzen 5 3600 using
`4096` worlds and `500` simulation steps:

| Path | Elapsed Time | Throughput |
| --- | ---: | ---: |
| Pure MuJoCo Warp physics with CUDA graph capture | 3.64 s | 562,518 world-steps/s |
| Physics with CPU-provided actions each step, no graph | 16.08 s | 127,396 world-steps/s |
| Current PPO rollout with optimizer epochs disabled | 24.19 s | 84,655 world-steps/s |
| Current PPO evaluator with PPO updates | 27.75 s | 73,796 world-steps/s |

During captured physics, GPU SM utilization reached approximately `84-100%`.
During the current PPO evaluator, it was generally approximately `23-34%`
while the process held roughly one CPU core busy. Disabling PPO optimizer
epochs saved only `3.56` seconds, so most elapsed time is not neural-network
optimization. It is rollout bridging and synchronization.

These timings are not performance predictions for a 96 GB GPU. They establish
the architectural bottleneck: a faster GPU will not be well utilized while
every timestep round-trips through CPU/NumPy.

## Current Bottlenecks

The critical loop is `train_ppo_policy()` in
`src/eureka_lite/mjwarp_evaluator.py`. Every simulation timestep currently:

1. Reads `data.qpos` and `data.qvel` from Warp device memory into NumPy.
2. Constructs policy observations on CPU.
3. Copies observations into Torch CUDA tensors.
4. Runs policy inference on CUDA.
5. Copies sampled actions and policy diagnostics back to CPU.
6. Creates/copies Warp action input from CPU to CUDA.
7. Advances MuJoCo Warp and synchronizes immediately.
8. Reads updated state to CPU again.
9. Computes generated reward components and true reward with NumPy.
10. Accumulates rollout buffers on CPU, then copies the PPO training batch back
    to CUDA.

There are two additional limitations:

- `compute_gae()` runs on NumPy arrays after each rollout chunk.
- `evaluate_policy_in_gym()` verifies a trained policy through serial CPU Gym
  episodes, instead of using the batched MJWarp path.

Candidate evaluation is also currently sequential in `search.py`. That is a
throughput opportunity, but it should not be addressed first: running multiple
inefficient host-synchronized evaluators at once risks increasing contention
without removing the central limitation.

## Behavioral Constraints

The refactor must preserve the scientific contract of the experiment:

- Each reward-code candidate trains an independent PPO actor-critic policy.
- Generated reward components affect policy training only.
- Verified reward remains the original Ant reward evaluated in the MJWarp
  target environment.
- Component-level shaped-reward statistics remain available for EUREKA
  feedback and audit records.
- Failed candidate programs and invalid generated code continue to produce
  negative RLVR samples under the existing option.
- Results and checkpoint formats remain readable or receive an explicit schema
  migration.

An optimization that changes the verified score, shares policy weights between
candidates, or silently weakens PPO training is not acceptable as a speedup.

## Target Rollout Architecture

### Device-Resident State And Buffers

Create a GPU rollout implementation in which the following stay on CUDA for an
entire PPO horizon:

- MuJoCo Warp state and controls;
- policy observations and sampled actions;
- action log probabilities and value estimates;
- shaped reward components and original Ant reward;
- done/active masks and episode accumulators;
- PPO rollout buffers;
- advantages and returns.

CPU transfer should occur only for compact summaries needed for logging,
checkpoint records, and final ranked results.

### Torch And Warp Interoperation

The policy remains a Torch actor-critic network. Physics remains MuJoCo Warp.
The integration boundary should use device-to-device interoperability rather
than NumPy staging:

- view Warp state arrays as Torch CUDA tensors, or use DLPack/interoperable
  wrappers supported by the installed Warp/Torch versions;
- write Torch CUDA actions into the Warp control array without a host copy;
- keep a single CUDA device and compatible streams, with an explicit
  synchronization policy at rollout boundaries.

The first implementation should prefer correctness and inspectable ownership
over aggressive capture. CUDA graph capture can be considered once all dynamic
reward and reset operations work without host transfers.

### GPU Reward Execution

Generated reward expressions are currently evaluated through NumPy. That is
incompatible with GPU-resident rollouts. Replace this with a compiled or
translated device reward representation.

Recommended first design:

1. Continue validating the reward AST against the existing allowed expression
   set.
2. Translate allowed arithmetic and reward-variable references to Torch tensor
   operations on CUDA.
3. Compute both named shaped-reward components and the fixed original Ant
   reward in Torch.
4. Preserve component reductions for EUREKA feedback, copying only summary
   scalars to CPU after a PPO chunk or candidate evaluation.

This is less invasive than generating Warp kernels dynamically, while retaining
policy and rollout values in Torch where PPO already executes.

### Termination And Reset Handling

Active masks, health termination checks, accumulated returns, and GAE terminal
masks should be CUDA tensors. Reset behavior requires explicit validation
because frequent device-side reset decisions can prevent CUDA graph capture.

The first optimized version may retain a synchronization/reset at completed
episode or rollout boundaries. It must remove synchronization from every
simulation timestep.

### Verified Evaluation

After PPO training, evaluate the deterministic policy with the original
`Ant-v5` reward through MJWarp rather than serial `gym.make()` environments.
Evaluation must:

- use the original reward equation, independent of generated shaping terms;
- run multiple evaluation seeds or initial states corresponding to
  `--eval-episodes`;
- record per-episode return, mean, and standard deviation in the same result
  structure;
- optionally report transfer against Gym `Ant-v5` on a fixed-seed sample;
- validate rank stability within MJWarp before a long RLVR run.

## Work Plan

### Phase 0: Benchmark Harness And Correctness Fixtures

Add a repeatable performance harness before changing the evaluator:

- benchmark pure captured physics, current PPO rollout, PPO without optimizer
  epochs, and final optimized PPO;
- record elapsed time, world-steps/s, peak CUDA memory, GPU utilization, and
  component/evaluation summaries;
- add deterministic tests for observation formation, original Ant reward,
  generated reward components, GAE, terminal masking, and deterministic policy
  action output.

Deliverable: a benchmark command and baseline record checked into `docs/` or
written as a run artifact.

### Phase 1: Torch CUDA Reward And GAE Path

Move computation that is independent of Warp interop first:

- implement Torch versions of Ant reward variables, shaped component
  evaluation, original reward, and GAE;
- compare Torch CPU results to the current NumPy implementation;
- compare Torch CUDA results to Torch CPU within an explicit numerical
  tolerance;
- retain the current rollout plumbing temporarily so failures are localized.

Deliverable: correctness-tested device-compatible reward and advantage
calculation.

### Phase 2: Device-Resident PPO Rollout

Replace per-step CPU staging in `train_ppo_policy()`:

- allocate rollout tensors on CUDA up front;
- obtain observations from device state without `qpos.numpy()` or
  `qvel.numpy()` inside the step loop;
- send policy actions to `data.ctrl` device-to-device;
- execute reward, terminal mask, episode accumulation, and buffer writes on
  CUDA;
- synchronize and transfer only summary values at PPO update/logging
  boundaries.

Keep the existing `search` evaluator unchanged as a fallback while the PPO
path is being validated.

Deliverable: a PPO evaluator whose steady-state timestep loop does not call
NumPy or make explicit per-step device synchronizations.

### Phase 3: GPU Verified Evaluation

Implement a batched MJWarp verifier for trained policies:

- deterministic policy rollout;
- original Ant return only;
- result-compatible per-episode metrics;
- optionally cross-check against Gym for seeded transfer-reporting runs.

Deliverable: an selectable verifier initially, then default it after
equivalence is established.

### Phase 4: Profiling And Scaling On The 96 GB GPU

Benchmark the optimized evaluator on the target GPU across world counts such
as `1024`, `2048`, `4096`, `8192`, and higher values supported by memory and
contact-buffer limits.

Measure:

- policy-evaluation elapsed time per candidate;
- GPU SM utilization and peak memory;
- PPO update time versus rollout time;
- return variance and training stability;
- end-to-end EUREKA generation duration.

Choose `--worlds-per-candidate` based on stable GPU saturation and learning
quality, not on maximum allocatable memory.

Deliverable: recommended production hyperparameters for the 96 GB run.

### Phase 5: Candidate Scheduling, Only If Needed

Once one candidate efficiently uses the GPU, determine whether concurrent
candidate policies are beneficial. Options are:

- retain sequential candidates if one `4096+` world evaluator saturates GPU
  compute;
- interleave a small number of independent candidates if policy networks leave
  material GPU headroom;
- batch candidate policies explicitly only if the complexity is justified by
  measured throughput gains.

This phase must preserve one independent PPO network per reward candidate and
the existing EUREKA lineage/results contract.

Deliverable: a scheduling decision supported by utilization and end-to-end
generation benchmarks.

## Proposed Configuration Surface

Do not expose low-level controls before the implementation is stable. The
likely user-facing options after validation are:

| Argument | Purpose |
| --- | --- |
| `--mjwarp-rollout-mode gpu` | Use the GPU-resident PPO rollout path. |
| `--mjwarp-rollout-mode host` | Retain the existing path for regression comparisons. |
| `--mjwarp-verified-evaluator mjwarp` | Use batched GPU true-return evaluation. |
| `--mjwarp-verified-evaluator gym` | Use serial Gym evaluation for equivalence checks. |
| `--candidate-workers N` | Optional later scheduling control, only after profiling. |

The optimized path should become the default only after it satisfies the
correctness and throughput gates below.

## Acceptance Criteria

### Correctness Gates

- Torch device reward components match the existing NumPy calculations on
  fixed test trajectories.
- Original reward and termination behavior match Gym `Ant-v5` on validation
  episodes within documented tolerance.
- GAE and PPO loss calculations match the existing implementation for fixed
  rollout tensors.
- Candidate result records still contain true verified return, shaped
  component summaries, PPO diagnostics, and RLVR metadata.
- Resume/checkpoint behavior remains valid for an optimized evaluator run.

### Performance Gates

For `4096` worlds and the existing PPO network on the available profiling GPU:

- no `numpy()` conversion of MuJoCo state or policy output inside the steady
  rollout timestep loop;
- no explicit `wp.synchronize()` per simulation timestep;
- at least `3x` higher PPO evaluator world-steps/s than the current measured
  `73,796` world-steps/s baseline, unless profiling identifies a new documented
  hardware bottleneck;
- materially increased sustained GPU utilization during rollout.

On the target 96 GB GPU:

- record throughput and memory scaling before choosing the final serious-run
  world count;
- complete a full one-generation `16`-candidate benchmark before starting a
  20-iteration RLVR run.

## Risks And Decisions

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Torch/Warp interop creates implicit synchronizations | GPU utilization remains low despite code changes | Profile each phase and test transfer-free step loops directly. |
| GPU reward translation accepts different expressions than current validation | Experiment semantics change | Keep the current AST validator as the source of allowed syntax and add equivalence tests. |
| MJWarp verified returns differ from Gym returns | Transfer to the Gym reference is uncertain | Report Gym diagnostics separately; use MJWarp rank stability for target-domain RLVR quality. |
| Larger world batches improve throughput but weaken PPO training dynamics | Fast results become scientifically poor | Track verified-return learning curves and variance during scaling tests. |
| Concurrent candidate scheduling competes with model generation or training memory | Out-of-memory or unstable pipeline behavior | Profile Ant evaluation separately and coordinate GPU residency with RLVR model phases before enabling workers. |

## Implementation Order

The implementation order should be:

1. Add reproducible benchmarks and numerical fixtures.
2. Implement GPU-compatible reward and GAE calculations.
3. Move the PPO rollout timestep loop to CUDA-resident buffers.
4. Add and validate batched MJWarp true-return evaluation.
5. Profile world-count scaling on the 96 GB GPU.
6. Consider candidate concurrency only after single-candidate utilization is
   measured.

This ordering removes the measured bottleneck first while keeping the verified
reward semantics and RLVR training records under test throughout the refactor.
