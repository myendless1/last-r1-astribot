from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SafetyLimits:
    xyz_min: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    xyz_max: tuple[float, float, float] = (1.0, 1.0, 2.0)
    max_translation_step: float = 0.15


def validate_action_chunk(actions: np.ndarray, *, current: np.ndarray | None = None, limits: SafetyLimits = SafetyLimits()) -> None:
    values = np.asarray(actions, np.float32)
    if values.ndim != 2 or values.shape[1] not in (10, 20): raise ValueError(f"Invalid action chunk shape {values.shape}")
    if not np.isfinite(values).all(): raise ValueError("Action chunk contains NaN/Inf")
    arm_starts = (0,) if values.shape[1] == 10 else (0, 10)
    for start in arm_starts:
        xyz = values[:, start : start + 3]
        if (xyz < np.asarray(limits.xyz_min)).any() or (xyz > np.asarray(limits.xyz_max)).any(): raise ValueError("Action outside workspace")
        if current is not None:
            current_xyz = np.asarray(current, np.float32)[start : start + 3]
            if np.linalg.norm(xyz[0] - current_xyz) > limits.max_translation_step:
                raise ValueError("First action translation exceeds safety limit")
        if len(xyz) > 1 and np.linalg.norm(np.diff(xyz, axis=0), axis=1).max(initial=0) > limits.max_translation_step:
            raise ValueError("Action translation step exceeds safety limit")
