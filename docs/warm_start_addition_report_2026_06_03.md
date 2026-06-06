# Warm Start Addition Report

Date: 2026-06-03

## Purpose

The warm-start addition was introduced to make the EUREKA/RLVR experiment more practical and more informative on Ant. In the original setup, every generated reward candidate trained a fresh Ant PPO policy from random initialization. That made candidate evaluation expensive and noisy: much of the per-candidate compute budget was spent learning basic survival and early locomotion rather than testing whether the generated reward expression produced better behavior.

The revised design starts each candidate from a shared MJWarp Ant policy that already demonstrates early locomotion under the original Ant reward. The generated EUREKA reward then fine-tunes that same base policy. This changes candidate evaluation from "can this reward bootstrap an Ant from scratch?" to "does this reward improve or damage an already moving Ant policy?" That better matches the experiment's goal: train the code model from a verifiable performance signal produced by MJWarp Ant behavior.

## Motivation

The full experiment asks a language model to generate reward code. Each candidate reward is evaluated by training an Ant policy with PPO in MuJoCo Warp, then scoring the resulting policy with the original MJWarp Ant return. That score becomes the RLVR reward assigned to the model completion that generated the candidate code.

Cold-start PPO was a poor fit for this loop for three reasons.

First, cold-start Ant training has a long early phase where policies mostly learn balance, survival, and rudimentary movement. During that phase, many different shaped rewards can look similarly weak because the agent has not yet reached the behavioral regime where reward-shaping details matter.

Second, the wall-clock cost is high. With 16 candidates, 4096 worlds per candidate, and multiple EUREKA generations per RLVR iteration, the experiment multiplies the PPO cost many times. Spending a large fraction of that cost repeatedly rediscovering basic locomotion is inefficient.

Third, the RLVR signal needs stable candidate rankings. If a candidate's score is dominated by whether random PPO initialization happened to find locomotion quickly, the language model receives a noisier update. A common warm start reduces this source of variance because every candidate begins from the same policy state.

## Implementation

Warm start is implemented through the PPO initialization path in `src/eureka_lite/mjwarp_evaluator.py`. The evaluator now supports two initialization modes:

- `base`: load a saved Ant actor-critic checkpoint and use it as the initial policy.
- `scratch`: initialize the actor-critic randomly, preserving the previous cold-start behavior.

The default mode is now `base`. The default checkpoint is:

```text
checkpoints/ant_mjwarp_warm_start_1500.pt
```

The checkpoint is bundled in the repository so a separate machine can clone the repo and run the 96 GB experiment without first generating a base policy. The checkpoint is small, approximately 598 KB, because it only stores the PPO actor-critic weights and metadata.

For single-policy training, `load_base_policy_into_single` loads the checkpoint into an `AntActorCritic`. For batched candidate training, `load_base_policy_into_batched` first loads the checkpoint into a normal `AntActorCritic`, then copies each linear layer and the action log-standard-deviation into every candidate slice of `BatchedAntActorCritic`. This means all candidates start from identical network parameters while still training independently afterward.

The policy architecture is:

- shared MLP backbone with hidden layers `256 -> 128 -> 64`
- ELU activations
- actor mean head from the final hidden layer to the Ant action dimension
- critic head from the final hidden layer to a scalar value
- learned diagonal Gaussian `log_std`
- tanh-squashed actions
- observation normalization buffers stored in the policy state

This architecture is used both for the single warm-start policy and for the batched per-candidate policy used in the large GPU run.

## Warm Start Training Runner

A dedicated staged warm-start runner was added in:

```text
src/eureka_lite/warm_start_gate_runner.py
```

and exposed through:

```text
scripts/run_rtx2070_warm_start_gate.sh
```

The runner trains a PPO policy on the original MJWarp Ant reward in repeated stages. After each stage, it evaluates the current policy against baseline policies and checks whether the configured gate has been surpassed. It writes:

- `base_policy.pt`: latest policy checkpoint
- `best_base_policy.pt`: best policy checkpoint seen so far
- `warm_start_gate_status.json`: current/final status
- `warm_start_gate_history.json`: per-stage history

The runner can stop after a maximum time budget or after passing the gate. A plateau-based stop condition exists as an optional argument, but the final workflow does not use it by default because the user wanted a run that trains for the requested budget or until the gate is passed.

## Stabilization Work

The first warm-start tests showed that simply training longer was not enough to make the process robust. The policy could plateau at a poor score, which raised the question of whether the original Ant reward was malformed. The result of the investigation was that the reward itself was not the main issue; the training setup needed more standard PPO stabilization.

The safer stabilization changes implemented for PPO were:

- observation normalization
- clipped value updates
- KL-based early stopping
- nonzero entropy regularization
- action standard deviation floor
- existing clipped policy objective, GAE, advantage normalization, tanh log-prob correction, and gradient clipping retained
- per-candidate gradient clipping retained for batched candidate training

These changes are compatible with EUREKA because they stabilize policy optimization without changing the semantic meaning of the generated reward expression. The reward candidate still determines the shaped reward used for PPO fine-tuning; the optimizer simply has a more reliable training process.

## Gate Selection

Two gate levels were explored.

The higher gate, around 2500 mean verified return, proved that the system could produce a much stronger Ant policy. One successful long run reached a best mean verified return of about 2616 after roughly 3.9 hours on the RTX 2070.

However, that level was considered too strong as the default warm start for EUREKA. If the base policy already locomotes very effectively, the generated reward candidates have less room to create meaningful early improvements. The concern was that EUREKA would mostly compare small perturbations around an already competent policy instead of discovering useful shaping gradients during the transition into stable locomotion.

The final bundled checkpoint therefore uses a lower gate of 1500. This was chosen because it confirms real locomotion but leaves substantial improvement headroom for reward search. It places the agent after the point where learning has clearly begun, but before the policy has consumed most of the available Ant performance improvements.

## Final 1500-Gate Result

The bundled checkpoint was produced by:

```text
runs/rtx2070_warm_start_gate_1500_2026_06_03
```

The run used:

- device: `cuda:0`
- worlds per candidate: `1024`
- stage policy iterations: `256`
- PPO horizon: `16`
- PPO epochs: `2`
- PPO minibatch size: `4096`
- learning rate: `3e-4`
- eval episodes: `8`
- verification steps: `1000`
- max hours: `8`
- gate minimum return: `1500`
- random-policy margin: `400`
- zero-policy margin: `400`

It stopped with `gate_surpassed` after one stage and about `872` seconds. The best mean verified return was approximately `1723.69`, with a standard deviation of approximately `11.78` across the 8 evaluation episodes.

The baseline comparisons were:

- base policy mean: `1723.69`
- random policy mean: `994.83`
- zero policy mean: about `1001.33`
- base minus random: about `728.86`
- base minus zero: about `722.36`

This result is useful for the EUREKA experiment because the policy is clearly better than trivial baselines but is still far below the stronger 2500+ policy. It should let generated rewards influence gait quality, stability, velocity, and energy tradeoffs during candidate fine-tuning.

## Integration With The 96 GB Run

The large GPU launcher now uses warm start by default:

```text
scripts/run_full_mjwarp_rlvr_96gb.sh
```

The default serious experiment uses:

- DeepSeek-Coder-V2-Lite-Instruct as the code model
- 20 RLVR iterations
- 16 candidates per iteration
- 3 EUREKA generations
- 4 elites
- 4096 MJWarp worlds per candidate
- 32 PPO fine-tuning iterations per candidate
- 32 common-seed verification episodes
- MJWarp verified evaluator
- conservative verified score as the RLVR reward: `mean_return - 0.25 * std_return`
- negative RLVR samples enabled
- bundled checkpoint: `checkpoints/ant_mjwarp_warm_start_1500.pt`

Cold start remains available with:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --cold-start
```

In cold-start mode, the launcher switches back to scratch initialization and uses the larger cold-start PPO budget.

## Expected Effect On The Experiment

Warm start should reduce wasted candidate evaluation compute, improve ranking stability, and make the RLVR signal more about reward-design quality than about random PPO bootstrap success. It also makes the experiment more practical on the RTX PRO 6000 96 GB target because the expensive part of the loop becomes candidate-specific reward fine-tuning rather than repeatedly learning the earliest Ant behaviors from scratch.

The main tradeoff is that warm start changes the scientific question slightly. It is no longer a pure from-scratch EUREKA reproduction. It is a reward-search and RLVR experiment in a fixed MJWarp Ant training regime where candidate rewards fine-tune a shared early-locomotion base policy. That tradeoff is intentional for this project because the core goal is not matching the public EUREKA paper exactly; the core goal is generating a stable, verifiable reward signal for training the code model.

The remaining risk is that some unusual reward candidates might only be useful from scratch and could be disadvantaged by starting from an already locomoting policy. For Ant locomotion, this is acceptable as a default because reasonable rewards should still improve or preserve forward movement, stability, and healthy behavior from the shared base. The cold-start option remains available for audits or comparisons.
