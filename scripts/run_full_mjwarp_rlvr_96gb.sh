#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct}"
RUN_ROOT="${RUN_ROOT:-runs/deepseek_lite_ant_mjwarp_rlvr}"
ITERATIONS="${ITERATIONS:-3}"
POPULATION="${POPULATION:-16}"
GENERATIONS="${GENERATIONS:-3}"
EUREKA_ELITES="${EUREKA_ELITES:-4}"
WORLDS_PER_CANDIDATE="${WORLDS_PER_CANDIDATE:-4096}"
MJWARP_EVALUATOR="${MJWARP_EVALUATOR:-ppo}"
MJWARP_EPISODE_STEPS="${MJWARP_EPISODE_STEPS:-500}"
MJWARP_TRAINING_EPISODE_HORIZON="${MJWARP_TRAINING_EPISODE_HORIZON:-1000}"
MJWARP_POLICY_ITERATIONS="${MJWARP_POLICY_ITERATIONS:-96}"
MJWARP_PPO_HORIZON="${MJWARP_PPO_HORIZON:-32}"
MJWARP_PPO_EPOCHS="${MJWARP_PPO_EPOCHS:-4}"
MJWARP_PPO_MINIBATCH_SIZE="${MJWARP_PPO_MINIBATCH_SIZE:-16384}"
MJWARP_PPO_LEARNING_RATE="${MJWARP_PPO_LEARNING_RATE:-3e-4}"
MJWARP_ELITE_FRAC="${MJWARP_ELITE_FRAC:-0.1}"
MJWARP_ROLLOUT_MODE="${MJWARP_ROLLOUT_MODE:-gpu}"
MJWARP_VERIFIED_EVALUATOR="${MJWARP_VERIFIED_EVALUATOR:-mjwarp}"
MJWARP_VERIFICATION_STEPS="${MJWARP_VERIFICATION_STEPS:-1000}"
MJWARP_VERIFIED_AUDIT_GYM="${MJWARP_VERIFIED_AUDIT_GYM:-0}"
MJWARP_VERIFIED_AUDIT_MAX_ABS_DIFF="${MJWARP_VERIFIED_AUDIT_MAX_ABS_DIFF:-}"
MJWARP_REWARD_BACKEND="${MJWARP_REWARD_BACKEND:-eager}"
MJWARP_BATCH_CANDIDATES="${MJWARP_BATCH_CANDIDATES:-1}"
MJWARP_CUDA_GRAPH="${MJWARP_CUDA_GRAPH:-1}"
INCLUDE_NEGATIVE_RLVR_SAMPLES="${INCLUDE_NEGATIVE_RLVR_SAMPLES:-1}"
NEGATIVE_RLVR_MARGIN="${NEGATIVE_RLVR_MARGIN:-1.0}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
RLVR_EPOCHS="${RLVR_EPOCHS:-1}"
RLVR_BATCH_SIZE="${RLVR_BATCH_SIZE:-1}"
RLVR_LEARNING_RATE="${RLVR_LEARNING_RATE:-5e-5}"
RLVR_MAX_LENGTH="${RLVR_MAX_LENGTH:-8192}"
MIN_GPU_MEMORY_MB="${MIN_GPU_MEMORY_MB:-90000}"
ALLOW_SMALL_GPU="${ALLOW_SMALL_GPU:-0}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
OVERWRITE_COLLECTION="${OVERWRITE_COLLECTION:-0}"
CHECKPOINT_RETENTION="${CHECKPOINT_RETENTION:-all}"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_full_mjwarp_rlvr_96gb.sh [options]

Options:
  --iterations N              RLVR sample/evaluate/train iterations.
  --run-root PATH             Output directory for the iterative run.
  --population N              Reward candidates per iteration.
  --generations N             EUREKA generations per RLVR iteration.
  --eureka-elites N           Ranked elites included in each refinement prompt.
  --worlds-per-candidate N    Ant worlds per reward candidate.
  --mjwarp-evaluator NAME     Evaluator: ppo or search. Default: ppo.
  --mjwarp-episode-steps N    PPO control steps per policy iteration. Default: 500.
  --mjwarp-training-episode-horizon N
                              Max steps before a training world resets. Default: 1000.
  --mjwarp-policy-iterations N
                              Policy iterations per candidate. Default: 96.
  --mjwarp-ppo-horizon N      PPO rollout horizon before each update.
  --mjwarp-ppo-epochs N       PPO optimization epochs per rollout batch.
  --mjwarp-ppo-minibatch-size N
                              PPO minibatch size.
  --mjwarp-ppo-learning-rate X
                              PPO learning rate.
  --mjwarp-rollout-mode NAME  gpu or host. Default: gpu.
  --mjwarp-verified-evaluator NAME
                              mjwarp or gym. Default: mjwarp.
  --mjwarp-verification-steps N
                              Verified rollout horizon. Default: 1000.
  --mjwarp-verified-audit-gym Compare MJWarp verified returns to Gym Ant-v5.
  --mjwarp-verified-audit-max-abs-diff X
                              Fail the audit above this per-episode difference.
  --mjwarp-reward-backend NAME
                              eager or compiled. Default: eager.
  --no-mjwarp-candidate-batching
                              Evaluate candidates sequentially instead of in one GPU batch.
  --no-mjwarp-cuda-graph     Disable physics CUDA-graph replay.
  --no-negative-rlvr-samples Do not train the model on invalid/failed reward programs.
  --negative-rlvr-margin X   Penalty below the worst successful candidate. Default: 1.0.
  --allow-small-gpu           Run even if GPU memory is below the 96 GB default.
  --no-smoke-test             Skip the small MJWarp smoke test.
  --force-train               Retrain adapters even if trainer_metrics.json exists.
  --overwrite-collection      Replace existing collection artifacts.
  --checkpoint-retention NAME all or latest RLVR epoch checkpoints. Default: all.
  -h, --help                  Show this help.

Environment variables with the same uppercase names are still supported as defaults.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --population)
      POPULATION="$2"
      shift 2
      ;;
    --generations)
      GENERATIONS="$2"
      shift 2
      ;;
    --eureka-elites)
      EUREKA_ELITES="$2"
      shift 2
      ;;
    --worlds-per-candidate)
      WORLDS_PER_CANDIDATE="$2"
      shift 2
      ;;
    --mjwarp-evaluator)
      MJWARP_EVALUATOR="$2"
      shift 2
      ;;
    --mjwarp-episode-steps)
      MJWARP_EPISODE_STEPS="$2"
      shift 2
      ;;
    --mjwarp-policy-iterations)
      MJWARP_POLICY_ITERATIONS="$2"
      shift 2
      ;;
    --mjwarp-training-episode-horizon)
      MJWARP_TRAINING_EPISODE_HORIZON="$2"
      shift 2
      ;;
    --mjwarp-ppo-horizon)
      MJWARP_PPO_HORIZON="$2"
      shift 2
      ;;
    --mjwarp-ppo-epochs)
      MJWARP_PPO_EPOCHS="$2"
      shift 2
      ;;
    --mjwarp-ppo-minibatch-size)
      MJWARP_PPO_MINIBATCH_SIZE="$2"
      shift 2
      ;;
    --mjwarp-ppo-learning-rate)
      MJWARP_PPO_LEARNING_RATE="$2"
      shift 2
      ;;
    --mjwarp-rollout-mode)
      MJWARP_ROLLOUT_MODE="$2"
      shift 2
      ;;
    --mjwarp-verified-evaluator)
      MJWARP_VERIFIED_EVALUATOR="$2"
      shift 2
      ;;
    --mjwarp-verification-steps)
      MJWARP_VERIFICATION_STEPS="$2"
      shift 2
      ;;
    --mjwarp-verified-audit-gym)
      MJWARP_VERIFIED_AUDIT_GYM=1
      shift
      ;;
    --mjwarp-verified-audit-max-abs-diff)
      MJWARP_VERIFIED_AUDIT_MAX_ABS_DIFF="$2"
      shift 2
      ;;
    --mjwarp-reward-backend)
      MJWARP_REWARD_BACKEND="$2"
      shift 2
      ;;
    --no-mjwarp-candidate-batching)
      MJWARP_BATCH_CANDIDATES=0
      shift
      ;;
    --no-mjwarp-cuda-graph)
      MJWARP_CUDA_GRAPH=0
      shift
      ;;
    --no-negative-rlvr-samples)
      INCLUDE_NEGATIVE_RLVR_SAMPLES=0
      shift
      ;;
    --negative-rlvr-margin)
      NEGATIVE_RLVR_MARGIN="$2"
      shift 2
      ;;
    --allow-small-gpu)
      ALLOW_SMALL_GPU=1
      shift
      ;;
    --no-smoke-test)
      RUN_SMOKE_TEST=0
      shift
      ;;
    --force-train)
      FORCE_TRAIN=1
      shift
      ;;
    --overwrite-collection)
      OVERWRITE_COLLECTION=1
      shift
      ;;
    --checkpoint-retention)
      CHECKPOINT_RETENTION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command "$PYTHON_BIN"
require_command nvidia-smi

GPU_MEMORY_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ -z "$GPU_MEMORY_MB" ]]; then
  echo "Could not read GPU memory with nvidia-smi." >&2
  exit 1
fi
if (( GPU_MEMORY_MB < MIN_GPU_MEMORY_MB )) && [[ "$ALLOW_SMALL_GPU" != "1" ]]; then
  cat >&2 <<EOF
Detected GPU memory: ${GPU_MEMORY_MB} MiB.
This script is configured for the full 96 GB experiment and requires at least
${MIN_GPU_MEMORY_MB} MiB by default.

To run anyway, set:
  ALLOW_SMALL_GPU=1

For smaller tests, also override WORLDS_PER_CANDIDATE, POPULATION, and
MJWARP_EPISODE_STEPS.
EOF
  exit 1
fi

log "GPU preflight"
nvidia-smi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
log "Upgrading pip tooling"
"$PY" -m pip install --upgrade pip setuptools wheel

log "Installing PyTorch"
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  "$PY" -m pip install torch --index-url "$TORCH_INDEX_URL"
else
  "$PY" -m pip install torch
fi

log "Installing eureka-lite with MuJoCo Warp support"
"$PY" -m pip install -e ".[mjwarp]"

log "Verifying CUDA, MuJoCo Warp, and package versions"
"$PY" - <<'PY'
import importlib.metadata as md
import torch
import warp as wp
import mujoco
import mujoco_warp

wp.init()
print("torch:", torch.__version__)
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("torch.cuda.device:", torch.cuda.get_device_name(0))
print("mujoco:", mujoco.__version__)
print("mujoco-warp:", md.version("mujoco-warp"))
print("warp-lang:", md.version("warp-lang"))
print("warp.cuda_available:", wp.is_cuda_available())
print("warp.devices:", [str(device) for device in wp.get_devices()])
if not torch.cuda.is_available() or not wp.is_cuda_available():
    raise SystemExit("CUDA is not available to both PyTorch and Warp.")
PY

if [[ "$RUN_SMOKE_TEST" == "1" ]]; then
  log "Running small MJWarp Ant smoke test"
  "$PY" -m eureka_lite.mjwarp_ant \
    --worlds 128 \
    --steps 10 \
    --warmup-steps 2 \
    --device cuda:0 \
    --action-mode random-once
fi

PIPELINE_ARGS=(
  -m eureka_lite.pipeline
  --model-id "$MODEL_ID"
  --run-root "$RUN_ROOT"
  --iterations "$ITERATIONS"
  --population "$POPULATION"
  --generations "$GENERATIONS"
  --eureka-elites "$EUREKA_ELITES"
  --worlds-per-candidate "$WORLDS_PER_CANDIDATE"
  --mjwarp-evaluator "$MJWARP_EVALUATOR"
  --mjwarp-episode-steps "$MJWARP_EPISODE_STEPS"
  --mjwarp-training-episode-horizon "$MJWARP_TRAINING_EPISODE_HORIZON"
  --mjwarp-policy-iterations "$MJWARP_POLICY_ITERATIONS"
  --mjwarp-ppo-horizon "$MJWARP_PPO_HORIZON"
  --mjwarp-ppo-epochs "$MJWARP_PPO_EPOCHS"
  --mjwarp-ppo-minibatch-size "$MJWARP_PPO_MINIBATCH_SIZE"
  --mjwarp-ppo-learning-rate "$MJWARP_PPO_LEARNING_RATE"
  --mjwarp-elite-frac "$MJWARP_ELITE_FRAC"
  --mjwarp-rollout-mode "$MJWARP_ROLLOUT_MODE"
  --mjwarp-verified-evaluator "$MJWARP_VERIFIED_EVALUATOR"
  --mjwarp-verification-steps "$MJWARP_VERIFICATION_STEPS"
  --mjwarp-reward-backend "$MJWARP_REWARD_BACKEND"
  --negative-rlvr-margin "$NEGATIVE_RLVR_MARGIN"
  --eval-episodes "$EVAL_EPISODES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --device cuda
  --trainer-algorithm grpo
  --trainer-epochs "$RLVR_EPOCHS"
  --trainer-batch-size "$RLVR_BATCH_SIZE"
  --trainer-learning-rate "$RLVR_LEARNING_RATE"
  --trainer-max-length "$RLVR_MAX_LENGTH"
  --checkpoint-retention "$CHECKPOINT_RETENTION"
)

if [[ "$FORCE_TRAIN" == "1" ]]; then
  PIPELINE_ARGS+=(--force-train)
fi
if [[ "$OVERWRITE_COLLECTION" == "1" ]]; then
  PIPELINE_ARGS+=(--overwrite-collection)
fi
if [[ "$INCLUDE_NEGATIVE_RLVR_SAMPLES" != "1" ]]; then
  PIPELINE_ARGS+=(--no-negative-rlvr-samples)
fi
if [[ "$MJWARP_BATCH_CANDIDATES" != "1" ]]; then
  PIPELINE_ARGS+=(--no-mjwarp-candidate-batching)
fi
if [[ "$MJWARP_CUDA_GRAPH" != "1" ]]; then
  PIPELINE_ARGS+=(--no-mjwarp-cuda-graph)
fi
if [[ "$MJWARP_VERIFIED_AUDIT_GYM" == "1" ]]; then
  PIPELINE_ARGS+=(--mjwarp-verified-audit-gym)
fi
if [[ -n "$MJWARP_VERIFIED_AUDIT_MAX_ABS_DIFF" ]]; then
  PIPELINE_ARGS+=(--mjwarp-verified-audit-max-abs-diff "$MJWARP_VERIFIED_AUDIT_MAX_ABS_DIFF")
fi

log "Starting full MJWarp EUREKA/RLVR pipeline"
"$PY" "${PIPELINE_ARGS[@]}"

log "Experiment complete"
printf 'Run root: %s\n' "$RUN_ROOT"
printf 'Pipeline state: %s\n' "$RUN_ROOT/pipeline_state.json"
