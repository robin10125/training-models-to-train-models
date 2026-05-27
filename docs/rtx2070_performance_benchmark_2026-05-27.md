# RTX 2070 MJWarp Performance Benchmark - 2026-05-27

## Scope

This benchmark measures the current candidate-batched MuJoCo Warp PPO
evaluator on the locally available GPU. It is intended to locate performance
bottlenecks, not to validate policy-learning quality or predict a 96 GB GPU
without target-device measurements.

Hardware and runtime:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2070 |
| VRAM | 8192 MiB |
| Driver | 595.58.03 |
| Compute capability | 7.5 |
| PyTorch | 2.11.0+cu130 |
| Warp | 1.13.0 |

All measured evaluator cases used generated Ant reward candidates, the GPU PPO
rollout path, batched MJWarp verification with one one-step episode, and
cached MJWarp kernels after a small warm-up. The short verification setting
keeps the measurement focused on policy-training rollout throughput.

## World Scaling

Configuration: `16` candidates, `32` control steps, `1` policy iteration,
`0` PPO optimization epochs, eager reward execution, CUDA graph enabled.

| Worlds Per Candidate | Total Worlds | Seconds | Control Transitions/s | Torch Peak Allocated |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 8,192 | 3.5365 | 74,125 | 95.7 MiB |
| 1,024 | 16,384 | 5.4467 | 96,258 | 176.9 MiB |
| 2,048 | 32,768 | 9.5959 | 109,273 | 342.2 MiB |
| 4,096 | 65,536 | 18.0555 | 116,150 | 672.0 MiB |

The `65,536`-world case showed sustained `89-98%` sampled SM utilization after
startup. On this GPU, the production batch shape is large enough to keep
MJWarp busy; it is not visibly throttled by the Python reward-dispatch loop.

## PPO And CUDA Graph Cost

Configuration: `16 x 4096` worlds, `32` control steps, eager rewards.

| Case | Seconds | Control Transitions/s | Torch Peak Allocated |
| --- | ---: | ---: | ---: |
| No PPO epochs, CUDA graph | 18.0555 | 116,150 | 672.0 MiB |
| No PPO epochs, no CUDA graph | 18.4541 | 113,642 | 672.1 MiB |
| 1 PPO epoch, CUDA graph | 18.1905 | 115,288 | 1,736.1 MiB |
| 4 PPO epochs, CUDA graph | 18.7415 | 111,899 | 1,736.1 MiB |

At this shape, four PPO optimization epochs added only about `3.8%` wall time
relative to rollout-only execution. Physics/control/reward rollout dominates
this RTX 2070 measurement. CUDA graph replay improved the short test by about
`2.2%`; it is worthwhile to retain, but is not the principal remaining speed
lever on this device.

A longer two-chunk production-shaped case (`64` control steps, `4` PPO epochs)
sustained `113,697` control transitions/s with sampled SM utilization generally
at `97-99%`. `nvidia-smi dmon` observed approximately `4.2-4.5 GiB` framebuffer
use during the PPO case; Torch's tabled allocation does not include all Warp
simulator allocations.

## Candidate Batching And Reward Dispatch

Holding total worlds fixed at `32,768`, with no PPO epochs:

| Candidate Layout | Seconds | Control Transitions/s |
| --- | ---: | ---: |
| `1 x 32768` | 9.1780 | 114,249 |
| `16 x 2048` | 9.4667 | 110,765 |

The cost of evaluating 16 distinct reward programs at fixed world count was
about `3%` in this measurement. This means the current per-candidate reward
dispatch is a reasonable future optimization target for a faster GPU, but is
not a major RTX 2070 bottleneck.

Candidate batching itself is important. For four candidates with `1024`
worlds each:

| Scheduling | Seconds | Control Transitions/s |
| --- | ---: | ---: |
| Batched | 1.9970 | 65,636 |
| Sequential | 5.0699 | 25,853 |

Batched evaluation was approximately `2.54x` faster in this small-shape test.
The default generation-level candidate batching should remain enabled.

## Compiled Reward Backend

Configuration: `16 x 1024` worlds, `32` steps, no PPO epochs.

| Reward Backend | Seconds | Control Transitions/s |
| --- | ---: | ---: |
| Eager | 5.4578 | 96,061 |
| Compiled, first invocation | 11.1774 | 46,906 |
| Compiled, repeated invocation | 5.5241 | 94,909 |

The first compiled invocation was about `2.05x` slower and emitted a Torch
Dynamo recompilation-limit warning from dynamically evaluated reward
expressions. Repeated invocation did not outperform eager execution. Keep
`--mjwarp-reward-backend eager` as the production default until generated
reward execution is represented in a compiler-friendly fused form.

## Runtime Estimate On This RTX 2070

Using the measured `113,697` control transitions/s from the longer full-batch
case as a rough evaluator-only throughput:

| Scope | Ant Control Transitions | Estimated Collection Time |
| --- | ---: | ---: |
| One EUREKA generation | 3,145,728,000 | 7.69 hours |
| `20` RLVR iterations x `3` EUREKA generations | 188,743,680,000 | 19.2 days |

This estimate excludes code-model sampling, GRPO updates, checkpoint I/O, and
run interruptions. It demonstrates why the serious experiment needs a much
faster GPU and target-device benchmarking.

## Implications For A 96 GB GPU

The RTX 2070 results do not prove that a high-performance 96 GB GPU will show
the same bottleneck. They establish:

1. Candidate batching is already beneficial and should remain enabled.
2. The current RTX 2070 is saturated at `16 x 4096`; increasing worlds here
   would not address CPU starvation.
3. The Python reward dispatch overhead is small on this GPU but can become
   material when physics kernels execute substantially faster.
4. The current compiled reward implementation is not a production
   optimization.

On the target GPU, first repeat the `16 x 4096` measurement with profiler
traces. If rollout SM utilization remains high, optimize experiment budget and
policy-learning quality before fusing reward dispatch. If utilization drops
while frequent small Torch reward/reset kernels dominate the trace, prioritize
fused candidate reward execution and device-side reset optimization.
