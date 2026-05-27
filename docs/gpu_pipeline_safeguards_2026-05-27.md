# GPU Pipeline Safeguards And Validation Changes

Timestamp: 2026-05-27 (America/Toronto)

## Purpose

This change set protects the scientific meaning of the Ant RLVR experiment
while permitting throughput-oriented operation on a large GPU. It is additive:
existing checkpoint records remain readable, eager reward execution remains the
reference path, and storage retention is opt-in.

## Training Episode Turnover

The GPU PPO evaluator previously kept a terminated Ant world inactive until the
end of a complete policy iteration. A policy that fell early therefore received
fewer useful training transitions, which could distort candidate ranking.

GPU PPO now records the terminal transition and immediately resets only the
finished worlds through MuJoCo Warp's public masked-reset interface. Training
rollout budget and episode lifetime are now distinct:

- `--mjwarp-episode-steps` controls transitions collected per PPO policy iteration.
- `--mjwarp-training-episode-horizon` controls episode timeout and defaults to
  `1000`, matching the verified Ant evaluation horizon.

The same masked-reset logic is applied to single-candidate and batched GPU PPO
paths. It is not applied to legacy host/search evaluation paths.

## Verified Return Audit

The accelerated batched MJWarp verifier is the default production verifier
because MJWarp Ant is the stated target environment. Its score is the RLVR
reward. Gym `Ant-v5` is an optional transfer-reference domain, rather than the
definition of correctness for this experiment.

`--mjwarp-verified-audit-gym` evaluates the trained policy through Gym using the
same episode seeds and records per-episode absolute differences in candidate
metadata. `--mjwarp-verified-audit-max-abs-diff X` optionally enforces a
transfer threshold. Audit mode requires a 1000-step MJWarp verification horizon
so reported comparisons use the same nominal horizon.

## Policy Budget Calibration

`python -m eureka_lite.calibrate_mjwarp` evaluates a fixed candidate population
at multiple PPO policy-iteration budgets and common seeds. It writes return
variance, top-candidate agreement, and rank correlation against the largest
budget. This command does not sample from or update the language model.

Use the calibration report before raising or lowering serious-run PPO budgets.
The intended decision criterion is stable candidate ordering, not maximum raw
throughput.

## Optional Reward Execution Backend

`--mjwarp-reward-backend eager` remains the reference generated-reward
implementation. `compiled` wraps each validated Torch reward program with
`torch.compile` and automatically falls back to the eager implementation if
compiled execution fails. This permits measurement without removing the
known-compatible path for arbitrary generated reward expressions.

## Resource And Storage Controls

Pipeline iteration summaries now include collection and RLVR training elapsed
time plus CUDA allocated, reserved, and peak allocated memory at phase
boundaries. The telemetry runs outside PPO inner loops and therefore avoids
introducing per-step host synchronization.

The RLVR trainer accepts `--checkpoint-retention all|latest`. `all` preserves
prior behavior. `latest` removes superseded epoch checkpoints only after the
new resumable checkpoint has been completely written, bounding disk growth for
multi-epoch runs without removing the active resume target.

## Recommended Serious-Run Procedure

1. Run a short MJWarp smoke run, optionally with `--mjwarp-verified-audit-gym`
   to record transfer to Gym.
2. Run the policy-budget calibration command over several common seeds.
3. Choose the smallest PPO budget with stable elite and rank behavior.
4. Start the long RLVR run with MJWarp verification and preserve any optional
   Gym transfer audit as secondary experimental metadata.
5. Enable `--mjwarp-reward-backend compiled` only after benchmarking it against
   eager behavior on representative generated rewards.
