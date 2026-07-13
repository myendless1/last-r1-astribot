#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-last_r1}"
export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

CONDA_ENV="${CONDA_ENV:-last-r1-astribot-sft}"
if [[ -f /media/damoxing/fileset/conda/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /media/damoxing/fileset/conda/bin/activate "${CONDA_ENV}"
fi

MODEL_PATH="${MODEL_PATH:-/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct}"
CHECKPOINT="${CHECKPOINT:-${CKPT_DIR:-}}"
FREEZE_VISION="${FREEZE_VISION:-true}"
DATASET_ROOT="${DATASET_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/centrifuge_multidrop-f2}"
OUTPUT_DIR="${OUTPUT_DIR:-/media/damoxing/ckp/last_r1_astribot_sft}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# Platform env: WORLD_SIZE = number of nodes, RANK = node rank (0-indexed).
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
LIMIT_EPISODES="${LIMIT_EPISODES:-}"
EXTRA_ARGS=()
TORCHRUN_ARGS=()

resolve_latest_checkpoint() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "${path}"
    return
  fi
  if [[ -f "${path}/config.json" ]]; then
    echo "${path}"
    return
  fi
  local latest=""
  latest="$(find "${path}" -maxdepth 1 -type d -name 'global_step_*' | sort -t_ -k3 -n | tail -1 || true)"
  if [[ -n "${latest}" ]]; then
    echo "${latest}"
    return
  fi
  echo "${path}"
}

if [[ -n "${CHECKPOINT}" ]]; then
  MODEL_PATH="$(resolve_latest_checkpoint "${CHECKPOINT}")"
fi

if [[ -n "${LIMIT_EPISODES}" ]]; then
  EXTRA_ARGS+=("data.limit_episodes=${LIMIT_EPISODES}")
fi

EXTRA_ARGS+=("model.freeze_vision=${FREEZE_VISION}")

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}"
export WANDB_DIR

EXTRA_ARGS+=("trainer.project_name=${WANDB_PROJECT}")
EXTRA_ARGS+=("trainer.wandb_mode=${WANDB_MODE}")
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  EXTRA_ARGS+=("trainer.logger=[console,wandb]")
else
  EXTRA_ARGS+=("trainer.logger=[console]")
fi

if [[ "${NNODES}" -eq 1 ]]; then
  TORCHRUN_ARGS+=(--standalone --nproc_per_node="${NPROC_PER_NODE}")
else
  TORCHRUN_ARGS+=(
    --nproc_per_node="${NPROC_PER_NODE}"
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )
fi

echo "[astribot-sft] model=${MODEL_PATH}"
if [[ -n "${CHECKPOINT}" ]]; then
  echo "[astribot-sft] checkpoint_dir=${CHECKPOINT} (resolved -> ${MODEL_PATH})"
fi
echo "[astribot-sft] freeze_vision=${FREEZE_VISION}"
echo "[astribot-sft] dataset=${DATASET_ROOT}"
echo "[astribot-sft] output=${OUTPUT_DIR}"
echo "[astribot-sft] wandb_enabled=${WANDB_ENABLED} project=${WANDB_PROJECT} mode=${WANDB_MODE}"
echo "[astribot-sft] wandb_dir=${WANDB_DIR}"
echo "[astribot-sft] nproc_per_node=${NPROC_PER_NODE}"
echo "[astribot-sft] nnodes=${NNODES} node_rank=${NODE_RANK}"
if [[ "${NNODES}" -gt 1 ]]; then
  echo "[astribot-sft] master=${MASTER_ADDR}:${MASTER_PORT}"
fi

STATS_PATH="${REPO_ROOT}/assets/astribot_dataset_statistics.json"
NEED_STATS=true
if [[ -f "${STATS_PATH}" ]]; then
  if python - "${STATS_PATH}" "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
dataset_root = str(Path(sys.argv[2]).resolve())
with stats_path.open("r", encoding="utf-8") as f:
    payload = json.load(f)
metadata = payload.get("__metadata__", {})
if metadata.get("dataset_root") != dataset_root:
    sys.exit(1)
PY
  then
    NEED_STATS=false
  else
    echo "[astribot-sft] dataset statistics source mismatch; recomputing..."
  fi
fi

if [[ "${NEED_STATS}" == "true" ]]; then
  echo "[astribot-sft] computing dataset statistics..."
  python "${REPO_ROOT}/scripts/compute_astribot_statistics.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${STATS_PATH}"
fi

torchrun "${TORCHRUN_ARGS[@]}" \
  -m verl.trainer.vla_fsdp_sft_trainer \
  model.partial_pretrain="${MODEL_PATH}" \
  data.dataset_root="${DATASET_ROOT}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
