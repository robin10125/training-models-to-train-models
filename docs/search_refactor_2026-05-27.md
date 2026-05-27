# Search Refactor and Correctness Review - 2026-05-27

## Purpose

The EUREKA search stage generates reward-code candidates, evaluates executable
candidates by training Ant policies, ranks verified target-environment returns,
and emits records for the RLVR model update. This stage must preserve
generation lineage and must not publish an RLVR reward until a generation has
been scored consistently.

## Implemented Structure

`search.py` is now the orchestration layer. Its loop advances one current
generation and calls focused modules:

- `search_types.py` owns result/configuration schemas and the normalized
  `MjwarpOptions` mapping.
- `search_state.py` owns the resumable generation state:
  `needs_population` or `evaluating`.
- `search_artifacts.py` owns checkpoint compatibility, result publication,
  RLVR serialization, and event/log artifacts.
- `search_feedback.py` owns penalties, ranking annotations, elite context, and
  evolution-feedback formatting.

The flat MJWarp fields remain in public configuration objects because current
CLI flags and existing checkpoints use them. Conversion between public config
layers now passes through `MjwarpOptions`, so a new MJWarp option has one
normalization point.

## Correctness Changes

### Generation-Boundary Resume

Previously an HF checkpoint saved after generation `N` carried an empty
candidate list for generation `N+1`. Resume interpreted the empty list as an
instruction to generate generation zero again. That evaluated stale,
incorrectly conditioned completions under a later generation index.

The checkpoint now persists a current `GenerationState`. At an HF generation
boundary it explicitly stores `needs_population` with index `N+1`; resume
therefore invokes generation `N+1` with the saved elites and evolution
feedback. Version-1 checkpoints are interpreted compatibly: an empty candidate
list before the configured final generation means `needs_population`.

### Atomic Batch Progress

MJWarp PPO candidate batching evaluates one generation-level group on the GPU.
The previous search loop rewrote results and checkpoints once for every member
of a completed GPU batch. The new loop records a batch's raw results in one
checkpoint transaction. Sequential evaluation retains candidate-sized
transactions because each result is independently complete.

### Canonical RLVR Publication

Raw evaluations are resumable checkpoint data, not training samples.
`rlvr_records.jsonl` is now rewritten only after generation finalization has:

1. assigned failed-evaluation penalties when enabled;
2. separated invalid code completions into negative RLVR records;
3. ranked executable candidates and annotated elites.

This prevents a paused run from exposing unpenalized or unranked records to the
trainer. The full pipeline now accepts a pause during a partially evaluated
generation without requiring a records file that cannot yet be canonical.

Invalid completions remain useful negative RLVR samples, but are never eligible
as EUREKA elites or mutation/refinement parents.

### GPU Feedback Diagnostic

Evolution feedback expected `best_true_return_in_population` from PPO search,
while the GPU path previously logged only the population mean. Both single and
batched GPU PPO paths now reduce the best internal target-environment return on
device and materialize that value with the existing end-of-run metric transfer.

## Review Against Earlier Problems

| Problem | Status | Resolution |
| --- | --- | --- |
| HF resume could generate generation-zero candidates at a later boundary | Fixed | Explicit persisted generation phase and legacy migration logic. |
| Batched GPU candidates caused per-candidate artifact rewrites | Fixed | One checkpoint commit per evaluated group and one canonical publication per finalized generation. |
| `run_search` mixed state, artifacts, configuration, and feedback | Fixed | Split into state, artifacts, types, and feedback modules. |
| MJWarp option propagation was repeated and easy to omit | Mitigated without breaking CLI/checkpoints | `MjwarpOptions` centralizes translation while public flat fields remain compatible. |
| Raw results could become stale RLVR records before penalties/final rank | Fixed | Canonical records are emitted only after finalization. |
| GPU feedback requested a metric the GPU summaries did not supply | Fixed | Device-side best true-return tracking added for single and batched PPO. |

## Remaining Design Constraint

The public configuration schemas still expose flat `mjwarp_*` fields. Removing
them entirely would require a checkpoint schema and CLI compatibility migration;
central normalization provides the maintainability benefit without invalidating
existing run commands or resumable experiment artifacts.
