# Base-Policy PPO Warm Start Plan

Timestamp: 2026-05-27 (America/Toronto)

## Purpose

The current Ant EUREKA/RLVR pipeline evaluates each generated reward program by
training a candidate-specific PPO policy from initialization. At the serious
default, one candidate receives:

```text
4096 worlds * 500 control steps * 96 policy iterations = 196,608,000 transitions
```

Most of that budget can be spent rediscovering basic Ant locomotion. A
base-policy warm start is a throughput optimization: pretrain one competent Ant
policy once with the original MJWarp Ant reward, then initialize candidate PPO
evaluations from that same checkpoint and use a shorter candidate-specific PPO
fine-tuning budget.

This changes the inner question from "can this reward train Ant from scratch?"
to "can this reward fine-tune a competent Ant policy into better behavior?".
That is acceptable for this project only if it is explicit, selectable, and
validated against from-scratch candidate rankings.

## Proposed Training Modes

Add a PPO initialization mode:

```bash
--mjwarp-ppo-init-mode scratch|base|lineage
```

Recommended meanings:

| Mode | Meaning | Main Use |
| --- | --- | --- |
| `scratch` | Current behavior: every candidate starts from the seeded default policy initialization. | Fidelity checks and public EUREKA-style comparisons. |
| `base` | Every candidate starts from the same pretrained original-reward Ant policy checkpoint. | Fast large-scale RLVR collection. |
| `lineage` | A refined candidate starts from its parent candidate's policy checkpoint; new roots start from scratch or the base policy. | Evolutionary search studies after base mode is validated. |

The first implementation should support `scratch` and `base`. `lineage` is
useful but introduces more checkpoint bookkeeping and should follow only after
the base-policy path is stable.

## Base Policy Pretraining

Add a standalone command that trains one PPO policy in MJWarp with the original
Ant reward:

```bash
python -m eureka_lite.pretrain_mjwarp_ant_policy \
  --output checkpoints/base_ant_mjwarp_policy.pt \
  --mjwarp-worlds 4096 \
  --mjwarp-episode-steps 500 \
  --mjwarp-policy-iterations 96 \
  --mjwarp-ppo-horizon 32 \
  --mjwarp-ppo-epochs 4 \
  --mjwarp-ppo-minibatch-size 16384 \
  --seed 0
```

The checkpoint should contain:

- actor-critic `state_dict`;
- observation normalization state, if normalization is added later;
- PPO optimizer state only if resuming pretraining, not necessarily for
  candidate evaluation;
- environment and PPO hyperparameters used to create the checkpoint;
- verified MJWarp Ant return for the final policy;
- code version metadata.

Candidate evaluation should load the actor-critic weights into each candidate
slot of `BatchedAntActorCritic` so every reward program receives the same
starting policy.

## Candidate Fine-Tuning Budget

For base warm starts, begin with:

```bash
--mjwarp-policy-iterations 32
```

This gives each candidate:

```text
4096 worlds * 500 control steps * 32 policy iterations = 65,536,000 transitions
```

Also test `20`, `24`, and `48` iterations in calibration. The decision should
be based on rank stability and elite agreement, not just wall time.

If the ranking correlation between `base` mode and `scratch` mode is strong,
the expected collection speedup is roughly proportional to the budget reduction:

| Candidate Iterations | Transition Budget | Expected Evaluator Speedup Versus 96 |
| ---: | ---: | ---: |
| 20 | 40,960,000 | 4.8x |
| 24 | 49,152,000 | 4.0x |
| 32 | 65,536,000 | 3.0x |
| 48 | 98,304,000 | 2.0x |

These are budget-based estimates. Actual end-to-end speedup will be lower once
code-model sampling, GRPO training, checkpoint I/O, and verification overhead
are included.

## RLVR Signal Invariants

Warm starts must not change the scalar reward assigned to a generated
completion. For a successful candidate:

```text
rlvr_reward = verified MJWarp Ant return of the final trained policy
```

The generated shaped reward remains only the inner PPO training objective. It
must not become the RLVR reward.

Every candidate in the same EUREKA generation must use:

- the same base checkpoint;
- the same MJWarp environment seed plan;
- the same PPO fine-tuning budget;
- independent candidate-specific policy parameters after initialization;
- independent optimizer state unless an explicit ablation proves otherwise.

This preserves within-generation comparison: the ranking should depend on the
candidate reward program, not on different starting policies.

## Validation Protocol

Before making `base` the serious-run default, run a paired calibration:

```bash
python -m eureka_lite.calibrate_mjwarp \
  --candidates 16 \
  --worlds 4096 \
  --seeds 0 1 2 \
  --budgets 20 24 32 48 96 \
  --compare-init-modes scratch base \
  --base-policy-checkpoint checkpoints/base_ant_mjwarp_policy.pt \
  --output runs/base_warm_start_calibration
```

The calibration should report:

- Spearman rank correlation between `base` and `scratch` verified returns;
- top-1 and top-k elite agreement;
- verified-return mean and variance by candidate;
- cases where `base` mode hides a reward that learns well from scratch;
- cases where `base` mode rewards a candidate that only preserves the initial
  policy and does not improve learning.

Decision rule:

1. Use `base` mode for serious RLVR if rank correlation and elite agreement are
   stable across seeds.
2. Keep periodic `scratch` audits during long runs.
3. Fall back to `scratch` for any study that claims comparability to public
   EUREKA-from-scratch results.

The previously suggested `>0.9` rank correlation should be treated as a target,
not an assumed property of this repository.

## Implementation Steps

1. Add `src/eureka_lite/pretrain_mjwarp_ant_policy.py`.
2. Add a reusable original-reward PPO training entry point in
   `mjwarp_evaluator.py` instead of duplicating the candidate-training loop.
3. Extend evaluator config with:

   ```text
   mjwarp_ppo_init_mode: scratch|base|lineage
   mjwarp_base_policy_checkpoint: optional path
   ```

4. Extend CLI and scripts:

   ```bash
   --mjwarp-ppo-init-mode scratch|base|lineage
   --mjwarp-base-policy-checkpoint PATH
   --pretrain-base-policy
   --skip-base-policy-pretrain
   ```

5. Add checkpoint loading into the single-candidate and batched candidate PPO
   paths. The batched path should copy the same weights into all candidate
   slots before fine-tuning begins.
6. Add metadata to every candidate record:

   ```text
   ppo_init_mode
   base_policy_checkpoint
   base_policy_verified_return
   base_policy_training_budget
   candidate_finetune_budget
   ```

7. Update `scripts/run_full_mjwarp_rlvr_96gb.sh` so serious runs can pretrain
   the base policy once, reuse it on resume, and avoid retraining it when the
   checkpoint already exists.
8. Add tests for:

   - missing checkpoint fails clearly in `base` mode;
   - checkpoint weights are replicated across batched candidate slots;
   - `scratch` mode behavior remains unchanged;
   - candidate metadata records the initialization mode;
   - resume does not retrain an existing base checkpoint unless requested.

## Default Recommendation

Keep repository defaults conservative until validation exists:

```text
default CLI mode: scratch
recommended serious large-GPU mode after calibration: base
```

For the 96 GB experiment, the practical target is:

```bash
--mjwarp-ppo-init-mode base \
--mjwarp-base-policy-checkpoint checkpoints/base_ant_mjwarp_policy.pt \
--mjwarp-policy-iterations 32
```

The README should present this as the fast serious-run path only after the
calibration command has been run and inspected.

## Risks

Base warm starts can bias the model toward rewards that are good local
fine-tuning objectives but weak from-scratch learning objectives. That is not
automatically bad for this project, because the stated target is scalable
MJWarp Ant performance as a verifiable reward. It does mean results should not
be described as exact public EUREKA reproduction.

A second risk is candidate collapse: many rewards may preserve the same strong
base behavior and produce returns that are too similar for GRPO. If that
happens, increase fine-tuning iterations, use harder initial-state variation,
or periodically include scratch-trained audit candidates in the RLVR batch.

The base policy itself must not leak into the prompt as a target to imitate
unless that is an intentional experiment. The language model should still be
asked to write reward code, not policy code.

## Summary

Base-policy PPO warm starts are likely the largest wall-time reduction available
without lowering the number of EUREKA candidates or worlds. The change is
scientifically defensible if it is exposed as a mode, recorded in metadata, and
validated with paired scratch-vs-base ranking calibration. The fastest serious
configuration should use one shared original-reward MJWarp base policy and
roughly `20-32` candidate fine-tuning iterations, with periodic from-scratch
audits to ensure the RLVR signal remains meaningful.
