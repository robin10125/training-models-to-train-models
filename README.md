# Training Models to Train Models

Small EUREKA-inspired RLVR experiment for local development.

The purpose of this experiment is to train the coding model that generates EUREKA reward code. Generated reward functions are treated as model completions; downstream RL performance is used as the verified reward signal; and those verified rewards are used to update the base coding model with LoRA/GRPO-style RLVR.

NVIDIA's EUREKA project uses an LLM to generate reward functions, trains policies in simulation, evaluates them, then iterates on the reward code. This repo keeps that loop small enough for local development and adds the model-training side:

1. Generate a population of dense reward candidates.
2. Train a PPO policy for each candidate.
3. Evaluate each policy using the real environment reward.
4. Store the generator prompt, completion tokens, old logprobs, and verified return as RLVR data.
5. Update the reward-code-generating base model from those records.

The default task is `Ant-v5` from Gymnasium/MuJoCo. Ant locomotion is a lightweight local analogue of EUREKA-style robot reward design: generated rewards train the agent, while true environment return verifies whether the generated reward helped.

## Setup

Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch first. For CUDA-enabled systems, choose the wheel that matches the local CUDA/driver setup. For example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If CUDA wheels fail, install CPU PyTorch instead:

```bash
pip install torch
pip install -r requirements.txt
```

For a fresh machine with a large NVIDIA GPU, the full MJWarp/RLVR setup and
experiment can be launched with:

```bash
./scripts/run_full_mjwarp_rlvr_96gb.sh
```

The script creates `.venv`, installs PyTorch and `mujoco-warp`, runs a small
CUDA/MJWarp smoke test, then launches the single-command training pipeline that
collects a 16-candidate x 4096-world Ant EUREKA batch and trains a GRPO LoRA
adapter from the resulting RLVR records. By default it requires a GPU with at
least 90000 MiB of VRAM. Override settings with environment variables, for
example:

```bash
OUTPUT_DIR=runs/my_96gb_run \
ADAPTER_OUTPUT_DIR=runs/my_96gb_adapter \
./scripts/run_full_mjwarp_rlvr_96gb.sh
```

## Run

```bash
python -m eureka_lite --task Ant-v5 --generations 3 --population 4 --timesteps 50000 --eval-episodes 5 --device auto
```

The script writes results to `runs/`, including:

- `results.json`: ranked candidate scores.
- `best_reward.py`: best discovered reward expression.
- `rlvr_records.jsonl`: one RLVR-ready record per evaluated reward candidate.
- `events.jsonl`: append-only progress events.
- `run.log`: human-readable progress log.
- `checkpoint.json`: resume state.
- `run_config.json`: run configuration.

## RLVR Experiment Notes

See [docs/rlvr_eureka_experiment.md](docs/rlvr_eureka_experiment.md) for the RLVR/EUREKA experiment assumptions. In short: the mock generator remains available for fast tests, the HF backend can use a real coding model, and the verified reward signal is true environment return.

The mock generator can be replaced with the HF backend:

```bash
python -m eureka_lite --task Ant-v5 --generator hf --population 1 --generations 1 --timesteps 0
```

Resume interrupted serious runs with:

```bash
python -m eureka_lite --resume --output-dir runs/example_run
```

Train a first RLVR LoRA adapter from collected records:

```bash
python -m eureka_lite.rlvr_trainer \
  --algorithm grpo \
  --records runs/example_run/rlvr_records.jsonl \
  --output-dir runs/example_adapter \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct
```

## Batched GPU Ant Simulation

The default PPO search path uses Gymnasium/SB3 environments. To exercise many
Ant physics worlds in parallel on an NVIDIA GPU, install the optional MuJoCo
Warp dependency:

```bash
pip install -e ".[mjwarp]"
```

Run a batched Ant simulation smoke test:

```bash
python -m eureka_lite.mjwarp_ant \
  --worlds 4096 \
  --steps 1000 \
  --device cuda:0 \
  --action-mode random-once
```

This command is a pure simulation benchmark. The main reward-search CLI also
supports `--sim-backend mjwarp`, which uses the same GPU physics path for a
batched Ant policy-search evaluator.

Run the full MJWarp-backed EUREKA/RLVR pipeline directly after setup:

```bash
python -m eureka_lite.pipeline \
  --task Ant-v5 \
  --model-id deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --collection-output-dir runs/deepseek_lite_ant_mjwarp_16x4096 \
  --adapter-output-dir runs/deepseek_lite_ant_mjwarp_grpo_adapter \
  --population 16 \
  --generations 1 \
  --worlds-per-candidate 4096 \
  --mjwarp-episode-steps 500 \
  --mjwarp-policy-iterations 4 \
  --mjwarp-elite-frac 0.1 \
  --eval-episodes 5 \
  --device cuda \
  --trainer-algorithm grpo \
  --trainer-epochs 1 \
  --trainer-batch-size 1 \
  --trainer-learning-rate 5e-5
```

The MJWarp evaluator trains a batched population of simple Ant policies with
the generated reward expression, then scores the best policy with true
Gymnasium Ant return for the RLVR record. The pipeline then trains the GRPO LoRA
adapter from those records and writes `pipeline_state.json`.
