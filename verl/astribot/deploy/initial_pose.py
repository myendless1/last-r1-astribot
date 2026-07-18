from __future__ import annotations

from math import isfinite
from typing import Sequence


GROUP_SIZES = (4, 7, 1, 7, 1, 2)
DEFAULT_INIT_JOINT_ACTION = (
    (0.5863, -1.1816, 0.5947, -0.0006),
    (0.2850, -0.3639, -1.2369, 1.6490, -0.3651, -0.0550, -0.2365),
    (0.0,),
    (-0.9200, -0.4720, 1.6216, 1.9225, 0.4911, 0.0491, 0.4409),
    (0.0,),
    (-0.0064, 0.8870),
)


def normalize_init_joint_action(value: Sequence[Sequence[float]]) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or len(value) != len(GROUP_SIZES):
        raise ValueError("init_joint_action must contain torso, left arm/gripper, right arm/gripper, and head")
    result = []
    for index, (group, size) in enumerate(zip(value, GROUP_SIZES)):
        if isinstance(group, (str, bytes)) or len(group) != size:
            raise ValueError(f"init_joint_action group {index} must contain {size} values")
        values = [float(item) for item in group]
        if not all(isfinite(item) for item in values):
            raise ValueError(f"init_joint_action group {index} contains NaN/Inf")
        result.append(values)
    return result


def default_init_joint_action() -> list[list[float]]:
    return normalize_init_joint_action(DEFAULT_INIT_JOINT_ACTION)
