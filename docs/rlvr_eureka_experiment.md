# RLVR EUREKA Experiment Notes

This project is currently a small EUREKA-style evaluation harness on `Ant-v5`, not a full RLVR training system for a coding model.

## Mocked Generator

The current generator is intentionally mocked. It emits Ant reward expressions from hand-written templates and mutations. It does not run an open-source coding model, sample tokens, record token IDs, record per-token log probabilities, or update model weights.

The mock generator exists to validate the expensive part of the workflow first:

1. Produce candidate reward code.
2. Train a downstream RL policy using that generated reward.
3. Evaluate the trained policy with the original environment reward.
4. Store candidate scores in a format that can later become RLVR training data.

This staged setup is useful because EUREKA-style rollouts are slow and noisy. The evaluator, storage format, and ranking logic should be stable before adding a trainable LLM policy.

## Verified Reward Signal

The verified reward signal is the true environment return.

Here, "true environment return" means the sum of the original `Ant-v5` environment rewards over an evaluation episode. During evaluation, the generated reward expression is not used as the score. It is used only during training of the downstream PPO agent.

The separation is:

1. The generator proposes reward code.
2. PPO trains a policy using the proposed reward code.
3. The trained policy is evaluated in the original environment.
4. The original environment return becomes the verified score for that generated reward code.

For RLVR, that verified score is the scalar reward that would later be assigned to the coding model completion that produced the reward code.

## Future LLM Training Data

When the mock generator is replaced with a real coding model, each rollout record should include:

- The exact prompt sent to the coding model.
- The sampled completion text.
- The sampled completion token IDs.
- The old per-token log probabilities under the model checkpoint that generated the completion.
- The model checkpoint or adapter identifier.
- The generated reward code.
- The downstream training configuration.
- The true environment return from evaluation.
- The evaluation status, seeds, and failure metadata.

Those records are sufficient for PPO- or GRPO-style updates without storing hidden states or KV caches.

## Current Run Status

The basic local experiment is runnable once the virtual environment is installed:

```bash
cd /home/robin/Downloads/eureka-lite
source .venv/bin/activate
python -m eureka_lite --task Ant-v5 --generations 1 --population 2 --timesteps 5000 --eval-episodes 2 --device auto
```

This smoke run should verify that candidate generation, PPO training, true-return evaluation, and output writing all work end to end.

Expected outputs are:

- `runs/latest/results.json`: ranked candidate scores.
- `runs/latest/best_reward.py`: best discovered reward expression.
- `runs/latest/rlvr_records.jsonl`: one RLVR-ready record per evaluated candidate.
- `runs/latest/rlvr_records.incremental.jsonl`: append-only records written after each candidate finishes.
- `runs/latest/events.jsonl`: append-only progress and lifecycle events.
- `runs/latest/run.log`: human-readable progress log.
- `runs/latest/checkpoint.json`: resume state for interrupted runs.
- `runs/latest/run_config.json`: immutable run configuration for reproducibility.

For a more meaningful local Ant run:

```bash
python -m eureka_lite --task Ant-v5 --generations 3 --population 4 --timesteps 50000 --eval-episodes 5 --device auto
```

## Remaining Work

The current implementation validates the evaluation loop, but it is not yet a complete RLVR system.

- The mock generator remains available for fast tests. The real generator backend uses `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`, a 16B-total / 2.4B-active MoE coding model, when run with `--generator hf`.
- The LLM policy update loop is implemented as a LoRA/QLoRA trainer over `rlvr_records.jsonl`. It supports GRPO-style clipped policy updates from stored old logprobs, plus a simpler weighted-SFT baseline.
- Short runs are smoke tests only. Meaningful Ant results need substantially more PPO timesteps per candidate.
- There is no baseline comparison report. Candidate scores are saved, but the script does not yet automatically compare against the default Ant reward or previous runs.
- There is no checkpoint/resume support. A stopped long run currently needs to restart from the beginning.
- Candidate evaluation is sequential. Larger EUREKA-style sweeps need parallel workers or a job queue.
- The reward-expression search space is intentionally small. It is enough to test plumbing, but not enough for a serious reward-code-generation benchmark.

The next practical step is to run the smoke command above. If it completes and writes all expected files, increase timesteps and population before adding real coding-model sampling.

## Real Generator Backend

The recommended replacement for the mock generator is:

- Model: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`
- Reason: smallest open-weight coding model with a defensible GPT-4-Turbo-class coding claim.
- Size: 16B total parameters, 2.4B active parameters, 128K context.
- Local target: single 24 GB GPU with 4-bit quantization.

Generate one validated reward candidate without PPO training:

```bash
python -m eureka_lite \
  --task Ant-v5 \
  --generator hf \
  --population 1 \
  --generations 1 \
  --timesteps 0 \
  --eval-episodes 0 \
  --model-id deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
```

Run one HF-generated candidate through the EUREKA evaluator:

```bash
python -m eureka_lite \
  --task Ant-v5 \
  --generator hf \
  --population 1 \
  --generations 1 \
  --timesteps 5000 \
  --eval-episodes 2 \
  --device cpu \
  --model-id deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
```

HF-generated records should include non-null `completion_token_ids` and `old_logprobs`. Hidden states and KV caches are still intentionally not stored.

## RLVR Trainer

Train a LoRA adapter from collected EUREKA records:

```bash
python -m eureka_lite.rlvr_trainer \
  --algorithm grpo \
  --records runs/hf_qwen3b_eureka_local/rlvr_records.jsonl \
  --output-dir runs/qwen3b_rlvr_adapter \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct \
  --epochs 1 \
  --batch-size 1 \
  --learning-rate 5e-5
```

The trainer:

- Reads successful records with non-null `verified_reward`.
- For `--algorithm grpo`, groups records by prompt and generator checkpoint, then computes group-relative clipped advantages.
- For `--algorithm grpo`, uses stored `old_logprobs` to compute a clipped policy-ratio loss.
- For `--algorithm weighted_sft`, normalizes verified rewards globally and weights completion-token loss.
- Masks prompt tokens and trains only on completion tokens.
- Saves a PEFT LoRA adapter and `trainer_metrics.json`.

Limitations:

- GRPO uses stored old logprobs and a clipped ratio objective, but does not yet evaluate a separate frozen reference model online.
- The current KL term is an approximate log-ratio penalty against the behavior policy logprobs.
- Use small learning rates and inspect generated rewards after each adapter update.

## Serious Run Infrastructure

Longer runs should use a unique output directory and can be resumed if interrupted:

```bash
python -m eureka_lite \
  --task Ant-v5 \
  --generator hf \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct \
  --population 3 \
  --generations 3 \
  --timesteps 20000 \
  --eval-episodes 5 \
  --device cpu \
  --temperature 0.4 \
  --top-p 0.9 \
  --max-new-tokens 128 \
  --output-dir runs/hf_qwen3b_eureka_local
```

Resume the same run after interruption:

```bash
python -m eureka_lite \
  --resume \
  --output-dir runs/hf_qwen3b_eureka_local
```

Use `--overwrite` only when intentionally replacing an existing run directory.

Failure handling:

- Candidate exceptions are captured as `status="failed"` records instead of terminating the whole run.
- `checkpoint.json` is updated after every candidate and generation.
- `events.jsonl` records run start, generation start/finish, candidate finish/skip, and run finish events.
- `run.log` mirrors important progress events in a human-readable format.
- `rlvr_records.incremental.jsonl` is append-only for recovery/debugging; `rlvr_records.jsonl` is the current canonical full snapshot.
