# Astribot protocol

The maintained Astribot path uses LeRobot 0.4.4 data converted from teleoperation HDF5 files.

## Vision

`base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb` are composed into one RGB T-layout frame. The same composition is used for current policy observations and future DINOv3 latent targets.

## Embodiment scopes

- `right_arm_gripper`: 10D, right xyz + rotation-6D + gripper.
- `dual_arm_gripper`: 20D, left and right xyz + rotation-6D + gripper.

Torso, head, and chassis dimensions remain in the raw 31D LeRobot record but do not participate in normalization, model input, action loss, or execution.

## Policy

The policy autoregressively predicts eight latent embeddings and then decodes the complete eight-step action chunk in parallel using bidirectional attention inside the action block.
