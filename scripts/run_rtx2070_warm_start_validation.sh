#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/rtx2070_warm_start_validation}"

"$PY" -m eureka_lite.warm_start_validation \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --population "${POPULATION:-4}" \
  --worlds-per-candidate "${WORLDS_PER_CANDIDATE:-512}" \
  --episode-steps "${EPISODE_STEPS:-128}" \
  --base-policy-iterations "${BASE_POLICY_ITERATIONS:-64}" \
  --candidate-policy-iterations "${CANDIDATE_POLICY_ITERATIONS:-8}" \
  --ppo-horizon "${PPO_HORIZON:-16}" \
  --ppo-epochs "${PPO_EPOCHS:-2}" \
  --ppo-minibatch-size "${PPO_MINIBATCH_SIZE:-4096}" \
  --verification-steps "${VERIFICATION_STEPS:-1000}" \
  --eval-episodes "${EVAL_EPISODES:-5}" \
  --acceptance-margin "${ACCEPTANCE_MARGIN:-25.0}" \
  "$@"
