#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"

: "${ASTRIBOT_RL_CHECKPOINT:?Set ASTRIBOT_RL_CHECKPOINT to an Astribot SFT/RL checkpoint}"
: "${ASTRIBOT_NORM:?Set ASTRIBOT_NORM to its right_arm_gripper normalization JSON}"
: "${ASTRIBOT_PROMPT:?Set ASTRIBOT_PROMPT to the exact task instruction}"

gateway_uri=${ASTRIBOT_GATEWAY_URI:-ws://127.0.0.1:8007}
num_gpus=${NUM_GPUS:-1}
output_dir=${OUTPUT_DIR:-outputs/astribot_real_rl}
mkdir -p "${output_dir}"
read -r action_dim action_horizon latent_length max_prompt_length dimension_scope < <(
  python - "${ASTRIBOT_RL_CHECKPOINT}/astribot_protocol.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    p = json.load(stream)
print(p["action_dim"], p["action_horizon"], p["latent_length"], p["max_prompt_length"], p["dimension_scope"])
PY
)
if [[ "${dimension_scope}" != right_arm_gripper || "${action_dim}" != 10 ]]; then
  echo "Real-robot RL currently requires a 10D right_arm_gripper checkpoint" >&2
  exit 2
fi

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
  hydra.run.dir="${output_dir}" \
  data.task_suite_name=astribot_real data.num_trials_per_task=1 \
  data.train_batch_size=1 data.val_batch_size=1 data.n_samples=1 \
  data.filter_accuracy=true data.accuracy_lower_bound=0 data.accuracy_upper_bound=1 \
  actor_rollout_ref.model.path="${ASTRIBOT_RL_CHECKPOINT}" \
  actor_rollout_ref.model.vla=qwen-oft \
  actor_rollout_ref.model.action_token_len="${action_dim}" actor_rollout_ref.model.action_chunks_len="${action_horizon}" \
  actor_rollout_ref.actor.action_token_len="${action_dim}" actor_rollout_ref.actor.action_chunks_len="${action_horizon}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  actor_rollout_ref.actor.traj_mini_batch_size=1 \
  actor_rollout_ref.actor.use_latent=true actor_rollout_ref.actor.use_latent_loss=true \
  actor_rollout_ref.actor.latent_length="${latent_length}" actor_rollout_ref.actor.latent_mode=ar \
  actor_rollout_ref.actor.latent_end_num=4 actor_rollout_ref.actor.latent_bind=0 \
  actor_rollout_ref.actor.input_mode=ids actor_rollout_ref.actor.value_choice=latent_end \
  actor_rollout_ref.actor.attn_mode=causal \
  actor_rollout_ref.rollout.task_suite_name=astribot_real \
  actor_rollout_ref.rollout.pretrained_checkpoint="${ASTRIBOT_RL_CHECKPOINT}" \
  actor_rollout_ref.rollout.data_status="${ASTRIBOT_NORM}" \
  actor_rollout_ref.rollout.action_token_len="${action_dim}" actor_rollout_ref.rollout.action_chunks_len="${action_horizon}" \
  actor_rollout_ref.rollout.max_prompt_length="${max_prompt_length}" actor_rollout_ref.rollout.latent_length="${latent_length}" \
  actor_rollout_ref.rollout.use_latent=true actor_rollout_ref.rollout.latent_mode=ar \
  actor_rollout_ref.rollout.latent_end_num=4 actor_rollout_ref.rollout.latent_bind=0 \
  actor_rollout_ref.rollout.input_mode=ids actor_rollout_ref.rollout.value_choice=latent_end \
  actor_rollout_ref.rollout.attn_mode=causal \
  actor_rollout_ref.rollout.use_proprio=true actor_rollout_ref.rollout.num_images_in_input=1 \
  actor_rollout_ref.rollout.micro_batch_size=1 actor_rollout_ref.rollout.val_micro_batch_size=1 \
  actor_rollout_ref.rollout.astribot_gateway_uri="${gateway_uri}" \
  actor_rollout_ref.rollout.astribot_prompt="${ASTRIBOT_PROMPT}" \
  critic.model.action_token_len="${action_dim}" critic.model.action_chunks_len="${action_horizon}" \
  trainer.n_gpus_per_node="${num_gpus}" trainer.nnodes=1 \
  trainer.default_local_dir="${output_dir}/checkpoints" "$@"
