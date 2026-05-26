# Command Line Reference

## Full Pipeline

Run iterative EUREKA collection followed by RLVR model updates:

```bash
python -m eureka_lite.pipeline [options]
```

### Experiment Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model-id` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | HF code model. |
| `--run-root` | `runs/deepseek_lite_ant_mjwarp_rlvr` | Output directory. |
| `--iterations` | `3` | Outer RLVR sample/evaluate/train iterations. |
| `--population` | `16` | Reward programs sampled per EUREKA generation. |
| `--generations` | `3` | EUREKA generations per RLVR iteration. |
| `--eureka-elites` | `4` | Top programs placed in refinement feedback. |
| `--seed` | `7` | Base random seed. |
| `--device` | `cuda` | Device: `auto`, `cpu`, or `cuda`. |

### Ant Evaluation Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `--worlds-per-candidate` | `4096` | Parallel Ant worlds trained for each reward program. |
| `--mjwarp-evaluator` | `ppo` | Policy optimizer: `ppo` or legacy `search`. |
| `--mjwarp-episode-steps` | `500` | World horizon for policy training. |
| `--mjwarp-policy-iterations` | `4` | Policy training iterations. |
| `--mjwarp-ppo-horizon` | `32` | PPO rollout horizon before an update. |
| `--mjwarp-ppo-epochs` | `4` | PPO optimization epochs per rollout batch. |
| `--mjwarp-ppo-minibatch-size` | `16384` | PPO minibatch size. |
| `--mjwarp-ppo-learning-rate` | `3e-4` | PPO learning rate. |
| `--mjwarp-elite-frac` | `0.1` | Elite fraction used only by legacy `search`. |
| `--eval-episodes` | `5` | Gym evaluation episodes used for verified return. |

### Generation and Negative Sample Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `--max-new-tokens` | `256` | Maximum reward-program completion length. |
| `--temperature` | `0.7` | HF generation temperature. |
| `--top-p` | `0.95` | HF nucleus sampling threshold. |
| `--no-4bit` | off | Disable 4-bit loading of the code model. |
| `--no-negative-rlvr-samples` | off | Do not train on invalid generations or failed evaluations. |
| `--negative-rlvr-margin` | `1.0` | Penalty below the worst successful verified return in a generation. |

### RLVR Trainer Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `--trainer-algorithm` | `grpo` | Update method: `grpo` or `weighted_sft`. |
| `--trainer-epochs` | `1` | LoRA training epochs per RLVR iteration. |
| `--trainer-batch-size` | `1` | Trainer batch size. |
| `--trainer-learning-rate` | `5e-5` | LoRA learning rate. |
| `--trainer-max-length` | `8192` | Maximum training sequence length; must cover full GRPO prompt context. |
| `--trainer-max-grad-norm` | `1.0` | Gradient clipping norm. |
| `--trainer-lora-r` | `16` | LoRA rank. |
| `--trainer-lora-alpha` | `32` | LoRA alpha. |
| `--trainer-lora-dropout` | `0.05` | LoRA dropout. |
| `--trainer-clip-epsilon` | `0.2` | GRPO clipping epsilon. |
| `--trainer-beta-kl` | `0.01` | GRPO KL coefficient. |
| `--overwrite-collection` | off | Replace collection output rather than resume it. |
| `--force-train` | off | Re-run adapter training even if metrics exist. |

## Bootstrap Script

The 96 GB GPU script creates the virtual environment, installs packages,
validates CUDA/MuJoCo Warp, runs a small simulator check, and launches the full
pipeline:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh [options]
```

Supported script options:

| Argument | Meaning |
| --- | --- |
| `--iterations N` | RLVR iterations. |
| `--run-root PATH` | Output directory. |
| `--population N` | Candidates per EUREKA generation. |
| `--generations N` | EUREKA generations per RLVR iteration. |
| `--eureka-elites N` | Ranked elites in refinement prompts. |
| `--worlds-per-candidate N` | Parallel Ant worlds for each candidate. |
| `--mjwarp-evaluator NAME` | `ppo` or `search`. |
| `--mjwarp-ppo-horizon N` | PPO rollout horizon. |
| `--mjwarp-ppo-epochs N` | PPO update epochs. |
| `--mjwarp-ppo-minibatch-size N` | PPO minibatch size. |
| `--mjwarp-ppo-learning-rate X` | PPO learning rate. |
| `--no-negative-rlvr-samples` | Disable invalid/failed model-update examples. |
| `--negative-rlvr-margin X` | Configure their penalty margin. |
| `--allow-small-gpu` | Bypass the 96 GB memory preflight. |
| `--no-smoke-test` | Skip the MuJoCo Warp startup check. |
| `--force-train` | Re-run adapter training. |
| `--overwrite-collection` | Replace prior collection data. |

The same script defaults can be set using uppercase environment variables,
including `ITERATIONS`, `POPULATION`, `GENERATIONS`, `EUREKA_ELITES`,
`WORLDS_PER_CANDIDATE`, `INCLUDE_NEGATIVE_RLVR_SAMPLES`, and
`NEGATIVE_RLVR_MARGIN`.

## Standalone Reward Search

Use the search command without the outer model-update loop:

```bash
python -m eureka_lite [options]
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--generations` | `3` | EUREKA search generations. |
| `--population` | `4` | Reward programs per generation. |
| `--eureka-elites` | `4` | Elite programs used as refinement context. |
| `--timesteps` | `10000` | SB3 training steps; a positive value also enables MJWarp evaluation. |
| `--eval-episodes` | `5` | Evaluation episodes. |
| `--n-envs` | `4` | Vector environments used by SB3. |
| `--seed` | `7` | Base seed. |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`. |
| `--sim-backend` | `sb3` | `sb3` or `mjwarp`. |
| `--worlds-per-candidate` | `4096` | MJWarp Ant worlds per candidate. |
| `--mjwarp-evaluator` | `ppo` | `ppo` or `search`. |
| `--mjwarp-episode-steps` | `500` | MJWarp policy horizon. |
| `--mjwarp-policy-iterations` | `4` | MJWarp policy iterations. |
| `--mjwarp-ppo-horizon` | `32` | PPO rollout horizon. |
| `--mjwarp-ppo-epochs` | `4` | PPO epochs. |
| `--mjwarp-ppo-minibatch-size` | `16384` | PPO minibatch size. |
| `--mjwarp-ppo-learning-rate` | `3e-4` | PPO learning rate. |
| `--mjwarp-elite-frac` | `0.1` | Legacy search elite fraction. |
| `--no-negative-rlvr-samples` | off | Exclude invalid/failed samples from output training records. |
| `--negative-rlvr-margin` | `1.0` | Negative sample penalty margin. |
| `--output-dir` | `runs/latest` | Collection output directory. |
| `--generator` | `mock` | `mock` or `hf`. |
| `--model-id` | DeepSeek Coder V2 Lite Instruct | HF model ID. |
| `--adapter-path` | none | Existing LoRA adapter used for generation. |
| `--max-new-tokens` | `256` | Completion token limit. |
| `--temperature` | `0.7` | Sampling temperature. |
| `--top-p` | `0.95` | Nucleus sampling threshold. |
| `--no-4bit` | off | Disable 4-bit HF loading. |
| `--resume` | off | Resume from `checkpoint.json`. |
| `--overwrite` | off | Replace existing collection artifacts. |

The standalone command produces RLVR records but does not train an adapter;
the full pipeline is the intended runner for automatic model updates.

## Standalone RLVR Trainer

Train an adapter from previously collected RLVR records:

```bash
python -m eureka_lite.rlvr_trainer --records PATH --output-dir PATH [options]
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--records` | required | Input `rlvr_records.jsonl` file. |
| `--output-dir` | required | Adapter and metrics output directory. |
| `--model-id` | `Qwen/Qwen2.5-Coder-3B-Instruct` | Base model for standalone training. |
| `--adapter-path` | none | Starting LoRA adapter. |
| `--max-length` | `8192` | Maximum sequence length; must include full GRPO prompt context. |
| `--epochs` | `1` | Training epochs. |
| `--batch-size` | `1` | Trainer batch size. |
| `--learning-rate` | `5e-5` | Learning rate. |
| `--max-grad-norm` | `1.0` | Gradient clipping norm. |
| `--lora-r` | `16` | LoRA rank. |
| `--lora-alpha` | `32` | LoRA alpha. |
| `--lora-dropout` | `0.05` | LoRA dropout. |
| `--no-4bit` | off | Disable 4-bit model loading. |
| `--algorithm` | `grpo` | `grpo` or `weighted_sft`. |
| `--clip-epsilon` | `0.2` | GRPO clipping epsilon. |
| `--beta-kl` | `0.01` | GRPO KL coefficient. |
| `--no-resume` | off | Start without an existing trainer checkpoint. |
| `--pause-path` | none | File whose presence pauses after an epoch. |
