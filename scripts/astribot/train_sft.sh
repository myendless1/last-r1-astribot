#!/usr/bin/env bash
set -euo pipefail

SCOPE="${SCOPE:-dual_arm_gripper}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG_NAME="${CONFIG_NAME:-${SCOPE}}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.astribot_sft_trainer \
  --config-name="${CONFIG_NAME}" \
  "$@"
