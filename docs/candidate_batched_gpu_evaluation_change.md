# Candidate-Batched GPU Evaluation Change

## Motivation

The previous MuJoCo Warp PPO path evaluated reward candidates one after another.
This preserved EUREKA semantics, but underused a large GPU: each candidate ran
only its own `4096` Ant worlds. The serious configuration contains `16`
candidates per EUREKA generation, so sequential execution repeatedly launches
small simulator workloads while leaving candidate-level parallelism unused.

The scientific constraint is that a larger GPU must not silently change the
learning problem. The experiment should still train one Ant policy from each
generated reward program, with `4096` worlds and the configured PPO budget for
that policy.

## New Execution Model

For the default `mjwarp` + `ppo` + `gpu` path, all executable reward candidates
in an EUREKA generation are now evaluated in one GPU job. With serious-run
defaults this creates:

```text
16 candidates * 4096 worlds per candidate = 65536 total simulated worlds
```

The flattened MuJoCo Warp state batch is reshaped as
`[candidate, world, state]` in Torch. Each candidate owns an independent
actor-critic parameter slice and optimizer gradients. A candidate's generated
reward program is applied only to that candidate's worlds. Verified returns and
RLVR records remain individual candidate results.

This is candidate batching, not pooling. It does not give one reward program
more Ant trajectories than another and does not replace the policy training
budget.

## Paired Inner-Loop Randomness

Candidates in one EUREKA generation retain the common evaluation seed. Batched
PPO initializes all candidate networks from identical initial parameter values
and supplies identical exploratory action noise for corresponding worlds.
Candidate policies then diverge only as their reward programs produce different
PPO updates. This makes within-generation ranking more directly attributable to
reward-code quality.

## Physics CUDA Graph Replay

The repeated five MuJoCo physics substeps for one `Ant-v5` control action are
now captured as a Warp CUDA graph after one warm-up step and state reset.
Torch policy operations run on Warp's CUDA stream so policy-written controls,
graph-replayed physics, and reward reads are ordered without per-step host
round trips.

Graph capture is best effort: metadata records `cuda_graph_requested` and
`cuda_graph_enabled`. If graph construction fails, training falls back to
ordinary MuJoCo Warp stepping instead of failing a candidate evaluation.

## Verification and Pausing

The batched MJWarp verified-return evaluator is the default because MJWarp Ant
is the experiment's target domain. Gym `Ant-v5` can be evaluated with
`--mjwarp-verified-audit-gym` as an optional transfer diagnostic; differences
between the two domains do not replace or invalidate the MJWarp RLVR signal.

When candidate batching is enabled, a pause request is observed after the
currently running candidate batch has completed and its individual records
have been checkpointed. With batching disabled, pause boundaries remain one
candidate at a time.

## Controls

These optimizations are enabled by default for the GPU PPO path:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --iterations 20
```

Disable candidate batching for an ablation or debugging run:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --no-mjwarp-candidate-batching
```

Disable physics graph replay independently:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --no-mjwarp-cuda-graph
```

The corresponding environment controls are
`MJWARP_BATCH_CANDIDATES=0` and `MJWARP_CUDA_GRAPH=0`.

## Validation

Implementation validation performed on an NVIDIA GeForce RTX 2070:

- the Python unit suite passes (`41` tests);
- a two-candidate CUDA/MuJoCo Warp PPO smoke evaluation completes;
- the smoke evaluation records `candidate_batching=true`,
  `total_batched_worlds=8`, and `cuda_graph_enabled=true`.

Earlier profiling motivated this implementation: a `4096`-world PPO workload
left substantial GPU utilization unused, while a `65536`-world diagnostic
batch materially increased aggregate transition throughput. A full `16 x 4096`
benchmark on the target 96 GB GPU is still required before reporting production
wall-clock estimates.
