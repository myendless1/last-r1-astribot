#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONDA_ENV="${CONDA_ENV:-last-r1-astribot-sft}"
if [[ -f /media/damoxing/fileset/conda/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /media/damoxing/fileset/conda/bin/activate "${CONDA_ENV}"
fi

CHECKPOINT="${CHECKPOINT:-/media/damoxing/ckp/last_r1_astribot_parallel_sft}"
DATASET_ROOT="${DATASET_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/centrifuge_multidrop-f3}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
START_FRAME="${START_FRAME:-0}"
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-}"
INFERENCE_MODE="${INFERENCE_MODE:-parallel}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
VIDEO_FPS="${VIDEO_FPS:-10}"
VIDEO_FRAME_STRIDE="${VIDEO_FRAME_STRIDE:-2}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
EVAL_DTYPE="${EVAL_DTYPE:-bf16}"

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --dataset-root "${DATASET_ROOT}"
  --episode-index "${EPISODE_INDEX}"
  --start-frame "${START_FRAME}"
  --inference-mode "${INFERENCE_MODE}"
  --video-fps "${VIDEO_FPS}"
  --video-frame-stride "${VIDEO_FRAME_STRIDE}"
  --eval-dtype "${EVAL_DTYPE}"
)

if [[ -n "${MAX_ROLLOUT_STEPS}" ]]; then
  ARGS+=(--max-rollout-steps "${MAX_ROLLOUT_STEPS}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ "${SAVE_VIDEO}" == "true" || "${SAVE_VIDEO}" == "1" ]]; then
  ARGS+=(--save-video)
fi

echo "[astribot-eval] checkpoint=${CHECKPOINT}"
echo "[astribot-eval] dataset=${DATASET_ROOT}"
echo "[astribot-eval] episode=${EPISODE_INDEX} start_frame=${START_FRAME}"
echo "[astribot-eval] inference_mode=${INFERENCE_MODE} eval_dtype=${EVAL_DTYPE} save_video=${SAVE_VIDEO}"

python "${REPO_ROOT}/scripts/eval_astribot_sft_rollout.py" "${ARGS[@]}" "$@"
