# Experiment Constitution

Created: 2026-05-28 (America/Toronto)

This document defines the core goals and non-negotiable design constraints for
this experiment. Any implementation change, optimization, refactor, new CLI
flag, or analysis pass must preserve these rules unless the experiment is
explicitly redefined.

## Core Goal

The goal of this experiment is to improve the reward generating capabilities of code models.
This is one part in a series of experiments to automate RLVR environment generation capabilities.

The experiment trains a code model to generate better reinforcement-learning
reward programs. A generated reward program is judged by downstream Ant policy
performance within the Eureka framework, not by static code quality or by the value of the generated reward
itself.

The central loop is:

1. A code model samples reward-code candidates.
2. EUREKA-style search ranks and refines those candidates across generations.
3. Each executable reward candidate trains an Ant PPO policy in MuJoCo Warp.
4. The trained policy is evaluated with the original MJWarp Ant return.
5. That verified return becomes the RLVR reward for the model completion.
6. GRPO/LoRA updates the code model, and the updated model samples the next
   iteration.

## Target Environment

MuJoCo Warp Ant is the target environment for this experiment.

Gym `Ant-v5`can be used for diagnostics, but is not the target environment.

## Verified Reward Invariant

For every successful candidate:

```text
rlvr_reward = verified MJWarp Ant return of the trained policy
```

The generated shaped reward must never be used as the final RLVR reward. It is
only the constant inner PPO training objective

Failure penalties for invalid code or failed evaluations may be used as
negative RLVR samples, but they must be recorded as penalties, not as measured
Ant performance.

## Candidate Independence

Each reward-code candidate must train its own Ant policy.

Candidate batching is allowed only when it preserves candidate independence:

- independent actor-critic parameter slices;
- independent optimizer gradients;
- candidate-specific generated reward programs;
- equal world count and PPO budget per active candidate;
- per-candidate verified returns and RLVR records.

Candidate batching must not pool worlds across candidates, share learned
policy updates across unrelated candidates, or give one reward program more
training data than another unless the run is explicitly marked as an ablation.

## EUREKA Search Invariant

The inner loop must remain EUREKA-style evolutionary reward search:

- prompt context includes relevant Ant/task/source information;
- prompt context must not expose the original Ant reward formula or verified
  reward implementation as an answer key;
- the base Ant task description is: "to make the ant run forward as fast as
  possible";
- candidates are ranked by verified target-environment return;
- elites and lower-ranked examples are available as feedback;
- reflection/evolution feedback is used for later generations;
- lineage, generation index, rank, elite status, and failure status are tracked.

Invalid completions may be negative RLVR samples, but they must not become
EUREKA elites or mutation/refinement parents.

## RLVR Training Invariant

The code model must be trained on the same sampled completion whose generation
log probabilities were recorded.

Records used for GRPO/RLVR must preserve:

- rendered prompt;
- raw completion;
- parsed reward components when available;
- token IDs and old log probabilities when available;
- verified return or explicit negative penalty;
- reward type;
- EUREKA lineage and rank metadata.

RLVR records must be published only after a generation has been finalized:
penalties assigned, candidates ranked, and elites annotated.

## PPO Budget And Seed Fairness

Within a comparable candidate group, candidates must share:

- the same world count;
- the same PPO policy-iteration budget;
- the same training episode horizon;
- the same verification horizon;
- paired seed plans for policy initialization, action sampling, and
  verification episodes.

Changing PPO budget, world count, seed strategy, or verification horizon is
allowed only when the run is clearly documented as a new configuration or
ablation.

## Warm-Start Constraint

Base-policy warm starts are the default operational mode for the serious
large-GPU experiment. They must remain explicit in command lines and metadata
and must not be described as identical to from-scratch EUREKA evaluation.

In `base` mode:

- every candidate in a comparable group must start from the same base policy
  checkpoint;
- the base policy checkpoint path and base policy metadata must be recorded;
- the candidate fine-tuning budget must be recorded;
- final RLVR reward remains verified MJWarp Ant return, not improvement over
  the base policy.

Warm-start runs answer a different question: whether a generated reward can
fine-tune a competent Ant policy. From-scratch runs answer whether a generated
reward can train Ant from initialization. These should not be conflated.

## Optimization Constraint

Performance work must preserve experimental meaning.

Allowed optimizations include:

- GPU-resident rollout buffers;
- Torch/Warp device interop;
- candidate-batched simulation with independent policies;
- CUDA graph replay;
- batched MJWarp verification;
- deferred metric synchronization;
- checkpoint retention controls.

Disallowed optimizations include:

- replacing verified return with shaped return;
- sharing policy gradients across unrelated candidates;
- using fewer transitions for some candidates without recording the run as an
  early-elimination or budget ablation;
- dropping failed or invalid completions silently;
- changing candidate ranking before generation finalization;
- hiding Gym/MJWarp transfer differences when a transfer audit is requested.

## Auditability Requirement

Every serious run must leave enough artifacts to reconstruct why a model update
happened:

- sampled reward source;
- generated completion;
- prompt context;
- validation or runtime errors;
- candidate lineage;
- verified returns;
- RLVR reward assignment;
- PPO/evaluator configuration;
- warm-start metadata when used;
- checkpoint/resume state.

If storage retention removes large intermediate artifacts, it must not remove
the active resume point or the records needed to audit model updates.

## Default Serious Configuration

The current default serious configuration is:

```text
task: Ant-v5 target behavior in MJWarp
candidates per EUREKA generation: 16
EUREKA generations per RLVR iteration: 3
elites per generation: 4
worlds per candidate: 4096
PPO init mode: base warm start
base-policy PPO iterations: 96
candidate PPO fine-tuning iterations: 32
control steps per policy iteration: 500
verified evaluator: MJWarp
outer trainer: GRPO with LoRA
```

The cold-start reference configuration is `--mjwarp-ppo-init-mode scratch` with
`--mjwarp-policy-iterations 96`. Smoke tests may reduce the PPO budget. All
budget and initialization changes must be explicit in command lines and
metadata.

## Change Review Checklist

Before accepting a change, answer these questions:

1. Does it preserve verified MJWarp Ant return as the RLVR reward?
2. Does each candidate still train an independent policy?
3. Are candidates in a comparison group given equal budget and paired seeds?
4. Are invalid and failed completions handled explicitly?
5. Are lineage, prompt, completion, rank, and reward metadata preserved?
