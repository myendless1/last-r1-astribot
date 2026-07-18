from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np


WEBXR_TO_ROBOT = np.asarray([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float64)
QUEST_FORWARD_XYZW = np.asarray([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)], np.float64)


def normalize_quat_xyzw(value: Any) -> np.ndarray:
    quat = np.asarray(value, np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], np.float64)
    return quat / norm


def quat_multiply_xyzw(a: Any, b: Any) -> np.ndarray:
    ax, ay, az, aw = normalize_quat_xyzw(a)
    bx, by, bz, bw = normalize_quat_xyzw(b)
    return normalize_quat_xyzw([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_inverse_xyzw(value: Any) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(value)
    return np.asarray([-x, -y, -z, w], np.float64)


def slerp_xyzw(a: Any, b: Any, alpha: float) -> np.ndarray:
    a, b = normalize_quat_xyzw(a), normalize_quat_xyzw(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        return normalize_quat_xyzw(a + alpha * (b - a))
    theta = math.acos(dot)
    return normalize_quat_xyzw((math.sin((1 - alpha) * theta) * a + math.sin(alpha * theta) * b) / math.sin(theta))


def quat_to_matrix_xyzw(value: Any) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(value)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def matrix_to_quat_xyzw(matrix: Any) -> np.ndarray:
    m = np.asarray(matrix, np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif i == 1:
            s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return normalize_quat_xyzw(q)


def controller_quat_to_robot(value: Any) -> np.ndarray:
    rotation = WEBXR_TO_ROBOT @ quat_to_matrix_xyzw(value) @ WEBXR_TO_ROBOT.T
    return quat_multiply_xyzw(matrix_to_quat_xyzw(rotation), QUEST_FORWARD_XYZW)


def webxr_delta_to_robot(value: Any, scale: float = 1.0) -> np.ndarray:
    x, y, z = np.asarray(value, np.float64).reshape(3)
    return np.asarray([-z, -x, y], np.float64) * float(scale)


def index_trigger_to_gripper(index: float, threshold: float = 0.2) -> float:
    fraction = np.clip((float(index) - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    return float((1.0 - fraction) * 100.0)


@dataclass
class QuestSample:
    active: bool = False
    delta_xyz: np.ndarray | None = None
    delta_quat_xyzw: np.ndarray | None = None
    gripper: float = 100.0
    outcome: str | None = None
    error: str = ""
    raw: dict[str, Any] | None = None


class QuestController:
    def __init__(self, state_url: str, *, timeout: float = 0.05, trigger_threshold: float = 0.5,
                 gripper_threshold: float = 0.2, verify_ssl: bool = False,
                 snapshot_provider: Callable[[], dict[str, Any]] | None = None):
        self.state_url = state_url
        self.timeout = float(timeout)
        self.trigger_threshold = float(trigger_threshold)
        self.gripper_threshold = float(gripper_threshold)
        self.verify_ssl = bool(verify_ssl)
        self.snapshot_provider = snapshot_provider
        self.position_anchor = None
        self.quat_anchor = None
        self._buttons = {4: False, 5: False}
        self._buttons_armed = False

    @property
    def enabled(self) -> bool:
        return bool(self.state_url or self.snapshot_provider)

    def reset(self) -> None:
        self.position_anchor = self.quat_anchor = None
        self._buttons = {4: False, 5: False}
        self._buttons_armed = False

    def _snapshot(self) -> dict[str, Any]:
        if self.snapshot_provider is not None:
            return self.snapshot_provider()
        import requests
        response = requests.get(self.state_url, timeout=self.timeout, verify=self.verify_ssl)
        response.raise_for_status()
        return response.json()

    def poll(self) -> QuestSample:
        if not self.enabled:
            return QuestSample()
        try:
            snapshot = self._snapshot()
        except Exception as exc:
            return QuestSample(error=str(exc))
        hand = snapshot.get("hands", {}).get("right", {})
        if not hand.get("valid", False):
            self.position_anchor = self.quat_anchor = None
            return QuestSample(raw=snapshot)
        buttons = hand.get("buttons", [])
        outcome = None
        current_buttons = {}
        for index, name in ((4, "success"), (5, "failure")):
            pressed = index < len(buttons) and float(buttons[index]) >= 0.5
            current_buttons[index] = pressed
            if self._buttons_armed and pressed and not self._buttons[index]:
                outcome = name
            self._buttons[index] = pressed
        if not self._buttons_armed and not any(current_buttons.values()):
            self._buttons_armed = True
        active = float(hand.get("middle", 0.0)) >= self.trigger_threshold
        if not active:
            self.position_anchor = self.quat_anchor = None
            return QuestSample(outcome=outcome, raw=snapshot)
        aligned = hand.get("aligned", {})
        position = np.asarray(aligned.get("position", [0, 0, 0]), np.float64)
        quat = controller_quat_to_robot(aligned.get("quaternion", [0, 0, 0, 1]))
        if self.position_anchor is None:
            self.position_anchor, self.quat_anchor = position.copy(), quat.copy()
        delta_xyz = webxr_delta_to_robot(position - self.position_anchor)
        delta_quat = quat_multiply_xyzw(quat_inverse_xyzw(self.quat_anchor), quat)
        return QuestSample(active=True, delta_xyz=delta_xyz, delta_quat_xyzw=delta_quat,
                           gripper=index_trigger_to_gripper(hand.get("index", 0.0), self.gripper_threshold),
                           outcome=outcome, raw=snapshot)
