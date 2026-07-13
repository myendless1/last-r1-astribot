from __future__ import annotations

from typing import Mapping

import numpy as np

STATE_KEY = "cartesian_so3_dict.cartesian_pose_state"
ACTION_KEY = "cartesian_so3_dict.cartesian_pose_command"
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
RAW_DIM = 31
RIGHT_ARM_GRIPPER = "right_arm_gripper"
DUAL_ARM_GRIPPER = "dual_arm_gripper"
DIMENSION_INDICES = {
    RIGHT_ARM_GRIPPER: tuple(range(19, 29)),
    DUAL_ARM_GRIPPER: tuple(range(9, 29)),
}
ALIASES = {"right": RIGHT_ARM_GRIPPER, "right_only": RIGHT_ARM_GRIPPER, "dual": DUAL_ARM_GRIPPER, "bimanual": DUAL_ARM_GRIPPER}


def normalize_dimension_scope(scope: str | None) -> str:
    normalized = str(scope or DUAL_ARM_GRIPPER).strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in DIMENSION_INDICES:
        raise ValueError(f"Unsupported Astribot dimension_scope={scope!r}; expected {sorted(DIMENSION_INDICES)}")
    return normalized


def get_astribot_dim(scope: str) -> int:
    return len(DIMENSION_INDICES[normalize_dimension_scope(scope)])


def select_astribot_dims(values: np.ndarray, scope: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1] != RAW_DIM:
        raise ValueError(f"Expected raw Astribot dimension {RAW_DIM}, got {array.shape}")
    return array[..., list(DIMENSION_INDICES[normalize_dimension_scope(scope)])]


def quaternion_xyzw_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    x, y, z, w = q
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return rotation[:2].reshape(-1).astype(np.float32)


def xyzquat_to_xyzrot6d(xyzquat: np.ndarray) -> np.ndarray:
    value = np.asarray(xyzquat, dtype=np.float32).reshape(-1)
    if value.shape != (7,):
        raise ValueError(f"Expected xyz+quaternion [7], got {value.shape}")
    return np.concatenate([value[:3], quaternion_xyzw_to_rotation6d(value[3:])])


def rotation6d_to_quaternion_xyzw(rotation6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation6d, dtype=np.float64).reshape(2, 3)
    row0 = value[0] / max(float(np.linalg.norm(value[0])), 1e-8)
    row1 = value[1] - np.dot(value[1], row0) * row0
    row1 = row1 / max(float(np.linalg.norm(row1)), 1e-8)
    row2 = np.cross(row0, row1)
    matrix = np.stack([row0, row1, row2])
    trace = float(np.trace(matrix))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        quat = np.array([(matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s, 0.25 * s])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quat = np.array([0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[2, 1] - matrix[1, 2]) / s])
        elif index == 1:
            s = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quat = np.array([(matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s, (matrix[0, 2] - matrix[2, 0]) / s])
        else:
            s = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quat = np.array([(matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s, (matrix[1, 0] - matrix[0, 1]) / s])
    return (quat / max(float(np.linalg.norm(quat)), 1e-8)).astype(np.float32)


def xyzrot6d_to_xyzquat(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.shape != (9,):
        raise ValueError(f"Expected xyz+rotation6d [9], got {value.shape}")
    return np.concatenate([value[:3], rotation6d_to_quaternion_xyzw(value[3:])])


def build_cartesian31(poses: Mapping[str, np.ndarray], joints: np.ndarray, frame_index: int) -> np.ndarray:
    joint = np.asarray(joints[frame_index], dtype=np.float32)
    if joint.shape[-1] < 22:
        raise ValueError(f"Expected at least 22 joint values, got {joint.shape}")
    tail = joint[-22:]
    return np.concatenate([
        xyzquat_to_xyzrot6d(poses["torso"][frame_index]),
        xyzquat_to_xyzrot6d(poses["left"][frame_index]), tail[11:12],
        xyzquat_to_xyzrot6d(poses["right"][frame_index]), tail[19:20], tail[20:22],
    ]).astype(np.float32)
