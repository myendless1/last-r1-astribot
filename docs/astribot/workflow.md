# Astribot workflow

The maintained path targets `lerobot==0.4.4`. Install `requirements-astribot.txt` without replacing the repository's vendored Transformers package.

## Prepare data

```bash
python scripts/astribot/convert_hdf5_to_lerobot.py \
  --raw-path /path/to/hdf5 --output-path /path/to/lerobot --task "task instruction"
python scripts/astribot/validate_lerobot_dataset.py --root /path/to/lerobot
```

Compute a separate normalization file for each trained embodiment. The horizon and frame stride must match training.

```bash
python scripts/astribot/compute_norm.py --root /path/to/lerobot \
  --output /path/to/norm_right.json --dimension-scope right_arm_gripper \
  --action-horizon 8 --action-frame-stride 1
```

Build future latent targets once. Every episode frame is encoded by DINOv3 once; future windows are assembled from those features.

```bash
python scripts/astribot/prepare_dino_latents.py --root /path/to/lerobot \
  --output /path/to/dino_latents --dino-path /path/to/dinov3 \
  --latent-length 8 --future-frame-stride 1 --batch-size 16
```

## Train and evaluate

Use either `right_arm_gripper.yaml` or `dual_arm_gripper.yaml`, and override all placeholder paths.

```bash
SCOPE=right_arm_gripper bash scripts/astribot/train_sft.sh \
  data.dataset_root=/path/to/lerobot \
  data.normalization_path=/path/to/norm_right.json \
  data.latent_cache_path=/path/to/dino_latents \
  model.partial_pretrain=/path/to/Qwen3-VL-4B-Instruct

python scripts/astribot/eval_open_loop.py --checkpoint /path/to/checkpoint \
  --dataset-root /path/to/lerobot --norm /path/to/norm_right.json \
  --latent-cache /path/to/dino_latents
```

The checkpoint protocol rejects mismatched scope, horizon, action stride, latent stride, prompt length, or image layout.

## Real robot

Start the policy server, then first run the client with `--dry-run`. The client performs one discarded warmup inference. For motion runs, it asks for confirmation unless `--non-interactive` is set.

```bash
python scripts/astribot/run_policy_server.py --checkpoint /path/to/checkpoint --norm /path/to/norm_right.json
python scripts/astribot/run_robot_client.py --prompt "task instruction" --dry-run
python scripts/astribot/run_robot_client.py --prompt "task instruction" \
  --init-hdf5 /path/to/reference_episode.hdf5 --execute-count 1
```

Only a short prefix of each predicted chunk is executed before observing again. Workspace, finite-value, first-step, and intra-chunk translation checks run before commands are sent.

## Real-robot RL and Quest takeover

Real-robot RL uses a separate Python 3.8 ROS/Astribot gateway. The gateway is
the only SDK command owner; the training process connects to it over WebSocket.
Start in observation-only mode and do not enable motion until images, state,
coordinates, workspace bounds, collision protection, and emergency stop have
been checked in an empty workspace.

```bash
# Robot shell (Python 3.8 / ROS Noetic)
pip install -r requirements-astribot-robot.txt
source /opt/ros/noetic/setup.zsh
source /opt/astribot_sdk/env.sh
python scripts/astribot/run_robot_gateway.py \
  --config configs/astribot_real_gateway.json --observation-only

# After editing the gateway config to set observation_only=false and
# robot_command_enabled=true, start the RL process in its training environment.
export ASTRIBOT_RL_CHECKPOINT=/path/to/checkpoint
export ASTRIBOT_NORM=/path/to/right_arm_gripper_norm.json
export ASTRIBOT_PROMPT='exact training task prompt'
bash scripts/astribot/run_real_rl.sh
```

The right Quest middle trigger preempts policy motion and streams right-arm
teleoperation commands. Right A marks success and right B marks failure. A
middle-trigger intervention invalidates the entire episode; human actions are
logged for audit but are never substituted for model tokens or included in PPO.
The model still predicts an eight-step chunk, while the gateway executes one
waypoint and re-observes. `finish_step` records actually executed waypoints and
`trajectory_steps` records model decisions for RL masking.

The optional Quest bridge can be started with
`QUEST_SERVER_ROOT=/path/to/quest/server scripts/astribot/quest_webxr.sh`.
This script checks ADB and applies `adb reverse`; it never installs udev rules
or performs privileged system changes.

## Legacy path

Root-level `*astribot_sft*` scripts and the v2.1 dataset modules describe the previous one-shot adaptation and are deprecated. The new trainer currently reuses its generic FSDP training loop, so those Python base modules remain in place; new experiments should use only `scripts/astribot/` and `verl/astribot/`.
