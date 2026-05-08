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
cd /home/robin/Downloads/eureka-lite
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for CUDA first. With your current NVIDIA driver, the CUDA 12.1 wheel is a reasonable starting point:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If CUDA wheels fail, install CPU PyTorch instead:

```bash
pip install torch
pip install -r requirements.txt
```

## Run

Quick smoke run:

```bash
python -m eureka_lite --task Ant-v5 --generations 1 --population 2 --timesteps 5000 --eval-episodes 2 --device auto
```

More useful local run:

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

See [docs/rlvr_eureka_experiment.md](docs/rlvr_eureka_experiment.md) for the current RLVR/EUREKA experiment assumptions. In short: the mock generator remains available for fast tests, the HF backend can use a real coding model, and the verified reward signal is true environment return.

The mock generator can be replaced with the HF backend:

```bash
python -m eureka_lite --task Ant-v5 --generator hf --population 1 --generations 1 --timesteps 0
```

Resume interrupted serious runs with:

```bash
python -m eureka_lite --resume --output-dir runs/your_run
```

Train a first RLVR LoRA adapter from collected records:

```bash
python -m eureka_lite.rlvr_trainer \
  --algorithm grpo \
  --records runs/your_run/rlvr_records.jsonl \
  --output-dir runs/your_adapter \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct
```

## Practical Limits

This is not a full reproduction of NVIDIA EUREKA. It intentionally avoids Isaac Gym and humanoid or dexterous-hand workloads. MuJoCo Ant is CPU-simulation-heavy, so `--device auto` or even `--device cpu` can be faster than forcing CUDA for small PPO runs.
