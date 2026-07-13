from __future__ import annotations

import numpy as np

from ..data.embodiment import DUAL_ARM_GRIPPER, RIGHT_ARM_GRIPPER, normalize_dimension_scope, xyzrot6d_to_xyzquat


def action_to_commands(action: np.ndarray, scope: str) -> list[list[float]]:
    scope = normalize_dimension_scope(scope); value = np.asarray(action, np.float32).reshape(-1)
    expected = 10 if scope == RIGHT_ARM_GRIPPER else 20
    if value.shape != (expected,): raise ValueError(f"Expected {expected}D action, got {value.shape}")
    if scope == RIGHT_ARM_GRIPPER:
        return [xyzrot6d_to_xyzquat(value[:9]).tolist(), np.clip(value[9:10], 0, 100).tolist()]
    return [xyzrot6d_to_xyzquat(value[:9]).tolist(), np.clip(value[9:10], 0, 100).tolist(),
            xyzrot6d_to_xyzquat(value[10:19]).tolist(), np.clip(value[19:20], 0, 100).tolist()]
