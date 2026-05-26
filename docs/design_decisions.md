# EUREKA and RLVR Design Decisions

## Objective

This project studies whether a code model can improve at designing reinforcement
learning reward programs when performance of the trained policy is used as a
verifiable reward. The task is `Ant-v5`: a generated reward program trains an
Ant controller, then the resulting controller is scored with the true
environment return.

There are two coupled optimization loops:

1. An inner EUREKA loop searches for better reward programs.
2. An outer RLVR loop updates the code model from the verified outcomes of its
   generated reward programs.

## EUREKA Loop

Each RLVR iteration contains multiple EUREKA generations. In each generation:

1. The code model samples a population of structured reward programs.
2. Each executable reward program trains one Ant PPO policy over parallel
   MuJoCo Warp worlds.
3. Policies are evaluated with true `Ant-v5` return.
4. Candidates are ranked by verified return and the configured number of elites
   is retained as context for the next generation.
5. The following generation receives source context, ranked elite reward
   programs, verified scores, lower-ranked examples, and policy diagnostics.

Serious Ant runs retain `4096` worlds per candidate, matching the default
actor count in public EUREKA's Ant configuration. The inner PPO budget is
`4096 * 500 * 96 = 196,608,000` control transitions per candidate, matching
the public EUREKA Ant transition count implied by `4096` actors, horizon `16`,
and `3000` policy iterations. This is a budget alignment rather than an exact
Isaac Gym implementation match.

Every executable candidate within an EUREKA generation is trained and verified
from the same seed. PPO also resets Torch RNG from that seed before policy
initialization and action sampling. GRPO therefore compares reward-program
outcomes under paired inner-loop randomness rather than assigning advantage for
different random policy initializations.

PPO policy training uses a GPU-resident MuJoCo Warp/Torch rollout path by
default. MuJoCo state, policy observations, actions, reward components, rollout
buffers, and GAE remain on device between PPO updates. A host rollout mode is
retained only for regression comparisons.

The default GPU PPO path evaluates all executable candidates in a generation
in one flattened MuJoCo Warp allocation. At the serious default this is
`16 * 4096 = 65536` simulated worlds, partitioned into `4096` worlds for each
independent candidate-specific PPO network. Candidates share initial network
values and corresponding exploration noise under the generation seed, but
their gradients and reward programs remain separate. Physics advances use a
best-effort CUDA graph for the five Ant frame-skip substeps. Both optimizations
can be disabled independently for ablations.

Prompts include pruned, reward-relevant source excerpts from Gymnasium `AntEnv`
(`step`, reward, observation, and reset code), the task adapter, and the local
MJWarp reward/evaluation path. This follows the EUREKA principle of exposing
environment code while avoiding unrelated simulator boilerplate that would
consume model and trainer context.

## Structured Reward Programs

The model is prompted to return a dictionary of named reward components, for
example:

```python
{"forward": "x_velocity", "healthy": "survive_reward", "control": "-0.01 * action_l2"}
```

The evaluator validates each component expression, sums the components to form
the policy-training reward, and records component-level statistics. The
reflection prompt can therefore show whether a candidate was dominated by
forward progress, survival, control penalties, or other terms.

Scalar reward expressions remain accepted for compatibility and are treated as
a single `total` component.

## Verified Reward and RLVR Reward

`verified_reward` means the true environment return obtained by the trained
policy. It is not replaced by the shaped reward or a failure penalty.

The default verified-return evaluator remains Gym `Ant-v5`, because it is the
reference environment contract. A batched MuJoCo Warp verifier is implemented
as an opt-in path for equivalence and scaling measurements; it must not replace
the Gym default solely because it is faster.

`rlvr_reward` is the scalar used to update the code model. For a successfully
evaluated reward program, it is the verified true environment return. For an
invalid generated program or a program that fails during evaluation, it is a
penalty below the worst successful program in the same generation:

```text
penalty = worst_successful_verified_reward - negative_rlvr_margin
```

If a generation contains no successful evaluations, the penalty is
`-negative_rlvr_margin`.

This split preserves experimental interpretability: failure penalties affect
model learning without being confused with measured Ant performance.

## Why Train on Invalid and Failed Programs

The model is asked to produce executable reward code. An invalid completion is
a verifiable failure of that task and should contribute a learning signal.
Likewise, code that passes parsing but fails evaluation is generally a negative
outcome for reward design.

Negative RLVR samples are enabled by default:

- `invalid_completion`: parsing or validation rejected a sampled program.
- `failed`: validation succeeded, but policy training or evaluation failed.

Each negative sample retains its prompt, raw sampled completion, token IDs, old
log probabilities, error text, EUREKA lineage, and assigned RLVR penalty.
Keeping the raw sampled completion is important because GRPO must update the
same token sequence whose old log probabilities were recorded.

The recorded prompt is the rendered chat-model input used during sampling, not
only the user-message fragment. GRPO also uses the original sampled completion
token IDs. To prevent an invalid importance-ratio calculation, GRPO raises an
error rather than truncating a prompt beyond `--trainer-max-length`; the full
pipeline defaults this limit to `8192` to accommodate source and reflection
context.

Negative samples may be disabled with `--no-negative-rlvr-samples`. This is
useful for ablations or if infrastructure failures would contaminate the
learning signal.

If a generation produces no executable candidates, collection terminates
cleanly rather than attempting policy evaluation. With negative samples
enabled, the outer trainer can still update the model from those invalid
completions.

## GRPO Grouping

GRPO compares completions generated from the same prompt and generator
checkpoint. Every EUREKA generation uses a shared prompt for its population, so
successful reward programs and negative completions from that shared prompt can
be compared within a group.

Retry prompts after a validator error include the validator feedback. Those
retry completions form their own prompt group. A retry contributes to GRPO only
when its group contains at least two trainable samples.

## Checkpointing and Auditability

Collection records contain:

- source and evolutionary prompt context;
- raw generated completion and generation log probabilities;
- parsed reward components and summed training expression;
- lineage, elite archive, generation rank, and selection status;
- verified true return;
- RLVR reward and reward type;
- validator or evaluation failure information;
- PPO/MJWarp diagnostics and component statistics.

Checkpoint files preserve the elite archive and evolutionary feedback used to
continue the search after resuming.

## Important Distinction

This project is not an exact reproduction of the original EUREKA experiment.
It combines EUREKA-style evolutionary reward discovery with MuJoCo Warp Ant
policy training and an outer GRPO/LoRA code-model update loop. The outer RLVR
training loop is the central experimental extension.
