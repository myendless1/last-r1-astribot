#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

CONDA_ENV="${CONDA_ENV:-last-r1-astribot-sft}"
if [[ -f /media/damoxing/fileset/conda/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /media/damoxing/fileset/conda/bin/activate "${CONDA_ENV}"
fi

MODEL_PATH="${MODEL_PATH:-/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/astribot_filter_300}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
PROFILE_STEPS="${PROFILE_STEPS:-20}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-10}"
LIMIT_EPISODES="${LIMIT_EPISODES:-20}"
TORCHRUN_ARGS=()

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

echo "[astribot-sft-profile] model=${MODEL_PATH}"
echo "[astribot-sft-profile] dataset=${DATASET_ROOT}"
echo "[astribot-sft-profile] profile_steps=${PROFILE_STEPS} warmup=${PROFILE_WARMUP_STEPS}"
echo "[astribot-sft-profile] limit_episodes=${LIMIT_EPISODES}"

if [[ ! -f "${REPO_ROOT}/assets/astribot_dataset_statistics.json" ]]; then
  python "${REPO_ROOT}/scripts/compute_astribot_statistics.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${REPO_ROOT}/assets/astribot_dataset_statistics.json"
fi

torchrun "${TORCHRUN_ARGS[@]}" \
  -m verl.trainer.vla_fsdp_sft_trainer \
  model.partial_pretrain="${MODEL_PATH}" \
  data.dataset_root="${DATASET_ROOT}" \
  data.limit_episodes="${LIMIT_EPISODES}" \
  trainer.profile_steps="${PROFILE_STEPS}" \
  trainer.profile_warmup_steps="${PROFILE_WARMUP_STEPS}" \
  trainer.default_local_dir="/tmp/astribot_vla_sft_profile" \
  "$@"
