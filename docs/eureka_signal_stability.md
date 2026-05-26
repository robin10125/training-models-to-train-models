# EUREKA Signal Stability Configuration

## Purpose

The outer RLVR update should teach the code model from differences in generated
reward programs, not from avoidable randomness in inner Ant policy training.
This project therefore uses an Ant evaluation configuration chosen for
comparability with public EUREKA and for a cleaner group-relative learning
signal.

## Worlds Per Candidate

Serious runs retain `4096` parallel Ant worlds per reward candidate. Public
EUREKA's Isaac Gym `AntGPT` task also uses `4096` actors by default. Keeping
this value avoids changing the experiment solely because a larger GPU can fit
more simulation state.

Larger batches such as `16384` worlds may be valuable as an ablation, but they
change policy optimization unless PPO minibatches and training duration are
controlled. They are not the default signal-generation setting.

## Policy Training Budget

The prior local default used:

```text
4096 worlds * 500 control steps * 4 policy iterations
  = 8,192,000 Ant control transitions per reward candidate
```

That budget is short for deciding whether one reward program truly produces a
better Ant controller. The serious-run default is now:

```text
4096 worlds * 500 control steps * 96 policy iterations
  = 196,608,000 Ant control transitions per reward candidate
```

This matches the transition count implied by the public EUREKA Ant setup:
`4096` actors, PPO horizon `16`, and `3000` RL policy iterations.
This project still uses its own MuJoCo Warp PPO implementation and defaults, so
the match is a training-transition budget comparison, not an exact replication
of Isaac Gym EUREKA.

For startup checks or pipeline validation, explicitly reduce the policy budget:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh \
  --iterations 3 \
  --mjwarp-policy-iterations 4
```

## Common Candidate Seeds

All executable reward candidates in the same EUREKA generation now receive the
same evaluation seed:

```text
candidate_seed = base_seed + generation * 100
```

The PPO evaluator resets Torch RNG from that seed before constructing the
policy and sampling stochastic actions. Final verified evaluation episodes also
derive from that shared seed. Within a generation, comparisons are therefore
paired across:

- policy initialization;
- PPO action sampling;
- verification episode seeds.

In the default GPU batched evaluator, this pairing is explicit: independent
candidate policy parameter slices are initialized identically and receive
common exploratory noise for corresponding Ant worlds. They diverge through
candidate-specific shaped rewards and PPO updates.

This removes avoidable candidate-ranking variance and improves the
interpretability of GRPO advantages. Different EUREKA generations still use
different seeds, preventing the full search from overfitting to one fixed
random trajectory set.

GPU simulation can still introduce small nondeterministic numerical effects.
For final conclusions, run top reward candidates under multiple independent
base seeds and report mean and variance of verified return.

## Recommended Serious Run

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh --iterations 20
```

The script defaults relevant to signal quality are:

| Setting | Default |
| --- | ---: |
| Reward candidates per EUREKA generation | `16` |
| EUREKA generations per RLVR iteration | `3` |
| Parallel Ant worlds per candidate | `4096` |
| Ant control steps per policy iteration | `500` |
| Policy iterations per candidate | `96` |
| Control transitions per candidate | `196,608,000` |
| GPU candidate batch | `16 * 4096 = 65536` worlds |
| Verified evaluator | Gym `Ant-v5` |

The verified Gym return remains the scalar success signal assigned to each
generated reward completion. The generated shaped reward is used to train its
Ant policy, but is not itself the RLVR score.
