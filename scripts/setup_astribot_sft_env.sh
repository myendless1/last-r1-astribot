#!/usr/bin/env bash
# Create a dedicated conda env for LaST-R1 astribot VLA SFT.
#
# Strategy:
#   1) Prefer cloning lingbot-va (fast, works on this cluster)
#   2) Fallback: fresh python=3.10 env + pip install (needs network/proxy)
set -euo pipefail

CONDA_ROOT="/media/damoxing/fileset/conda"
ENV_NAME="${ENV_NAME:-last-r1-astribot-sft}"
SOURCE_ENV="${SOURCE_ENV:-lingbot-va}"
USE_PROXY="${USE_PROXY:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${USE_PROXY}" == "1" ]]; then
  export https_proxy="${https_proxy:-http://192.168.32.28:18000}"
  export http_proxy="${http_proxy:-http://192.168.32.28:18000}"
  PIP_INDEX=(-i https://pypi.org/simple)
else
  PIP_INDEX=(-i https://pypi.tuna.tsinghua.edu.cn/simple)
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/bin/activate" base

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] env ${ENV_NAME} already exists"
else
  if conda env list | awk '{print $1}' | grep -qx "${SOURCE_ENV}"; then
    echo "[setup] cloning ${SOURCE_ENV} -> ${ENV_NAME}"
    conda create --name "${ENV_NAME}" --clone "${SOURCE_ENV}" -y
  else
    echo "[setup] creating fresh env ${ENV_NAME} (python=3.10)"
    conda create -n "${ENV_NAME}" python=3.10 -y -c defaults
  fi
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/bin/activate" "${ENV_NAME}"

echo "[setup] pinning LaST-R1 local transformers compatibility..."
pip install "${PIP_INDEX[@]}" \
  'tokenizers>=0.22.0,<=0.23.0' \
  'huggingface-hub>=0.34.0,<1.0' \
  codetiming \
  tensordict \
  'ray>=2.10.0' \
  decord \
  hydra-core \
  omegaconf \
  timm

# Optional: align torch with LaST-R1 README if the clone is too new/old.
if [[ "${INSTALL_TORCH_251:-0}" == "1" ]]; then
  echo "[setup] installing torch==2.5.1 cu124..."
  pip install "${PIP_INDEX[@]}" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python - <<'PY'
from transformers import AutoProcessor
import pandas, pyarrow, decord, torchvision, tensordict, hydra, omegaconf, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
model_path = "/media/damoxing/ckp/qwen_ft/Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
print("env_gate OK, tokenizer_len=", len(processor.tokenizer))
PY

python "${REPO_ROOT}/scripts/smoke_astribot_sft.py"

echo "[setup] done."
echo "  source ${CONDA_ROOT}/bin/activate ${ENV_NAME}"
echo "  export PYTHONPATH=${REPO_ROOT}:\$PYTHONPATH"
