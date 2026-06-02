#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/rtx2070_warm_start_gate}"

"$PY" -m eureka_lite.warm_start_gate_runner \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --seed "${SEED:-2070}" \
  --worlds-per-candidate "${WORLDS_PER_CANDIDATE:-256}" \
  --episode-steps "${EPISODE_STEPS:-128}" \
  --stage-policy-iterations "${STAGE_POLICY_ITERATIONS:-256}" \
  --ppo-horizon "${PPO_HORIZON:-16}" \
  --ppo-epochs "${PPO_EPOCHS:-2}" \
  --ppo-minibatch-size "${PPO_MINIBATCH_SIZE:-4096}" \
  --ppo-learning-rate "${PPO_LEARNING_RATE:-3e-4}" \
  --init-std "${INIT_STD:-0.35}" \
  --eval-episodes "${EVAL_EPISODES:-8}" \
  --verification-steps "${VERIFICATION_STEPS:-1000}" \
  --max-hours "${MAX_HOURS:-12}" \
  --plateau-hours "${PLATEAU_HOURS:-1}" \
  --plateau-tolerance "${PLATEAU_TOLERANCE:-25}" \
  --gate-min-return "${GATE_MIN_RETURN:-2500}" \
  --gate-random-margin "${GATE_RANDOM_MARGIN:-1000}" \
  --gate-zero-margin "${GATE_ZERO_MARGIN:-1000}" \
  "$@"
