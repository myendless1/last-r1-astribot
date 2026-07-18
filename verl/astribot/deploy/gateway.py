from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np

from ..data.embodiment import (
    IMAGE_KEYS,
    RIGHT_ARM_GRIPPER,
    get_astribot_dim,
    rotation6d_to_quaternion_xyzw,
    select_astribot_dims,
    xyzquat_to_xyzrot6d,
)
from .action_adapter import action_to_commands
from .initial_pose import default_init_joint_action, normalize_init_joint_action
from .quest import QuestController, normalize_quat_xyzw, quat_multiply_xyzw, slerp_xyzw
from .robot_io import read_images, read_state31
from .sdk_loader import load_astribot_class


GATEWAY_PROTOCOL_VERSION = 1


@dataclass
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8007
    sdk_root: str = ""
    robot_type: str = "S1"
    dimension_scope: str = RIGHT_ARM_GRIPPER
    quest_state_url: str = "https://127.0.0.1:8443/api/state"
    quest_poll_rate_hz: float = 100.0
    quest_control_rate_hz: float = 100.0
    quest_timeout: float = 0.05
    quest_verify_ssl: bool = False
    execute_count: int = 1
    waypoint_duration: float = 0.3
    camera_timeout: float = 0.3
    max_policy_translation_step_m: float = 0.04
    max_policy_rotation_step_deg: float = 15.0
    max_takeover_translation_step_m: float = 0.01
    max_takeover_rotation_step_deg: float = 2.5
    right_min_z: float | None = 0.862
    right_xyz_low: tuple[float, float, float] | None = None
    right_xyz_high: tuple[float, float, float] | None = None
    initial_joint_duration: float = 4.0
    init_joint_action: list[list[float]] = field(default_factory=default_init_joint_action)
    robot_command_enabled: bool = False
    observation_only: bool = False
    fake: bool = False
    require_start_confirmation: bool = True
    log_dir: str = "outputs/astribot_real_logs"

    def __post_init__(self) -> None:
        if self.dimension_scope != RIGHT_ARM_GRIPPER:
            raise ValueError("The first real-robot release only supports right_arm_gripper")
        self.init_joint_action = normalize_init_joint_action(self.init_joint_action)
        if self.execute_count < 1:
            raise ValueError("execute_count must be positive")


class _FakeRobot:
    torso_name, arm_left_name, arm_right_name = "torso", "left_arm", "right_arm"
    effector_left_name, effector_right_name, head_name = "left_gripper", "right_gripper", "head"
    whole_body_names = ["chassis", "torso", "left_arm", "left_gripper", "right_arm", "right_gripper", "head"]

    def __init__(self):
        self.state = np.zeros(31, np.float32)
        self.state[3:9] = [1, 0, 0, 0, 1, 0]
        self.state[12:18] = [1, 0, 0, 0, 1, 0]
        self.state[22:28] = [1, 0, 0, 0, 1, 0]
        self.state[19:22] = [0.4, 0.0, 0.9]
        self.state[18] = self.state[28] = 100
        self.initial_state = self.state.copy()
        self.calls = []

    def activate_camera(self, _config): pass
    def set_head_follow_effector(self, _enabled): pass
    def get_images_dict(self):
        images = {name: np.zeros((48, 64, 3), np.uint8) for name in ("Bolt", "left_D405", "right_D405")}
        return images, {}, {}, {}
    def move_joints_position(self, names, commands, **kwargs): self.calls.append(("reset", names, commands, kwargs))
    def move_cartesian_pose(self, names, commands, **kwargs): self.calls.append(("cartesian", names, commands, kwargs))
    def set_different_type_command(self, names, types, commands, **kwargs): self.calls.append(("mixed", names, types, commands, kwargs))
    def reset_state(self): self.state = self.initial_state.copy()


class AuditWriter:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "gateway.jsonl"
        self.items: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=32)
        self.thread = threading.Thread(target=self._run, daemon=True, name="astribot-audit-writer")
        self.thread.start()

    def submit(self, item: dict[str, Any]) -> None:
        try: self.items.put_nowait(item)
        except queue.Full: pass

    def _run(self) -> None:
        while True:
            item = self.items.get()
            if item is None: return
            images = item.pop("_images", None)
            if images:
                image_name = f"episode-{item.get('episode', 0):05d}-{time.time_ns()}.npz"
                np.savez_compressed(self.directory / image_name, **images)
                item["image_file"] = image_name
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    def close(self) -> None:
        try: self.items.put(None, timeout=1)
        except queue.Full: return
        self.thread.join(timeout=2)


class AstribotGateway:
    def __init__(self, config: GatewayConfig, *, robot=None, quest: QuestController | None = None):
        self.config = config
        if config.fake:
            self.robot = robot or _FakeRobot()
        else:
            Astribot = load_astribot_class(config.sdk_root or None)
            self.robot = robot or Astribot(freq=max(config.quest_control_rate_hz, 100), high_control_rights=True)
        if hasattr(self.robot, "set_head_follow_effector"):
            self.robot.set_head_follow_effector(False)
        if hasattr(self.robot, "activate_camera"):
            self.robot.activate_camera({"left_D405": {"flag_getdepth": False}, "right_D405": {"flag_getdepth": False}, "Bolt": {"flag_getdepth": False}})
        self.quest = quest or QuestController(config.quest_state_url, timeout=config.quest_timeout,
                                               verify_ssl=config.quest_verify_ssl)
        self.command_lock = threading.RLock()
        self.quest_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.intervened_event = threading.Event()
        self.terminated = False
        self.success = False
        self.infrastructure_error = ""
        self._takeover_anchor: np.ndarray | None = None
        self._last_takeover_target: np.ndarray | None = None
        self._latest_quest_error = ""
        self._episode_id = 0
        self.audit = AuditWriter(config.log_dir)
        self.quest_thread = threading.Thread(target=self._quest_loop, daemon=True, name="astribot-quest-control")
        self.quest_thread.start()

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": GATEWAY_PROTOCOL_VERSION,
            "dimension_scope": self.config.dimension_scope,
            "state_dim": get_astribot_dim(self.config.dimension_scope),
            "action_dim": get_astribot_dim(self.config.dimension_scope),
            "camera_names": list(IMAGE_KEYS),
            "image_color_order": "bgr",
            "sdk_ready": self.robot is not None,
            "quest_ready": self.quest.enabled and not bool(self._latest_quest_error),
            "robot_command_enabled": self.config.robot_command_enabled and not self.config.observation_only,
        }

    def _state31(self) -> np.ndarray:
        if isinstance(self.robot, _FakeRobot): return self.robot.state.copy()
        return read_state31(self.robot)

    def observe(self) -> dict[str, Any]:
        started = time.monotonic()
        images = read_images(self.robot)
        elapsed = time.monotonic() - started
        if self.config.camera_timeout > 0 and elapsed > self.config.camera_timeout:
            raise TimeoutError(f"Astribot camera read took {elapsed:.3f}s (limit {self.config.camera_timeout:.3f}s)")
        return {"images": images, "state": self._state31(), "image_color_order": "bgr", "time": time.time()}

    def reset(self, *, confirm: bool = True) -> dict[str, Any]:
        self._episode_id += 1
        self.terminated = self.success = False
        self.infrastructure_error = ""
        self.intervened_event.clear()
        with self.quest_lock:
            self.quest.reset()
        if isinstance(self.robot, _FakeRobot):
            self.robot.reset_state()
        if confirm and self.config.require_start_confirmation and not self.config.fake:
            input("Scene clear? Press Enter to reset Astribot and begin the episode (Ctrl-C to abort): ")
        if self.config.robot_command_enabled and not self.config.observation_only:
            with self.command_lock:
                self.robot.move_joints_position(self.robot.whole_body_names[1:], self.config.init_joint_action,
                                                duration=self.config.initial_joint_duration, use_wbc=False)
        try:
            observation = self.observe()
        except Exception as exc:
            return self._failure(f"Observation: {type(exc).__name__}: {exc}")
        self.audit.submit({"event": "reset", "episode": self._episode_id, "time": time.time()})
        return self._response(observation=observation)

    @staticmethod
    def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        dot = abs(float(np.dot(normalize_quat_xyzw(a), normalize_quat_xyzw(b))))
        return math.degrees(2 * math.acos(np.clip(dot, 0.0, 1.0)))

    def _validate_target(self, target: np.ndarray, current: np.ndarray, *, takeover: bool) -> None:
        if target.shape != (10,) or not np.isfinite(target).all():
            raise ValueError(f"Expected finite 10D right-arm action, got {target.shape}")
        distance = float(np.linalg.norm(target[:3] - current[:3]))
        max_distance = self.config.max_takeover_translation_step_m if takeover else self.config.max_policy_translation_step_m
        if distance > max_distance + 1e-6:
            raise ValueError(f"Translation step {distance:.4f}m exceeds {max_distance:.4f}m")
        angle = self._quat_angle_deg(rotation6d_to_quaternion_xyzw(target[3:9]),
                                     rotation6d_to_quaternion_xyzw(current[3:9]))
        max_angle = self.config.max_takeover_rotation_step_deg if takeover else self.config.max_policy_rotation_step_deg
        if angle > max_angle + 1e-4:
            raise ValueError(f"Rotation step {angle:.2f}deg exceeds {max_angle:.2f}deg")
        if self.config.right_min_z is not None and target[2] < self.config.right_min_z:
            raise ValueError(f"Right arm z={target[2]:.4f} below {self.config.right_min_z:.4f}")
        for bound, op, label in ((self.config.right_xyz_low, np.less, "lower"), (self.config.right_xyz_high, np.greater, "upper")):
            if bound is not None and op(target[:3], np.asarray(bound)).any():
                raise ValueError(f"Right arm target exceeds {label} workspace bound")

    def _send_right_target(self, target: np.ndarray, *, duration: float, streaming: bool) -> None:
        commands = action_to_commands(target, RIGHT_ARM_GRIPPER)
        if not (self.config.robot_command_enabled and not self.config.observation_only): return
        if hasattr(self.robot, "set_different_type_command"):
            self.robot.set_different_type_command(
                [self.robot.arm_right_name, self.robot.effector_right_name], ["cartesian", "joints"], commands,
                control_way="filter", use_wbc=False,
            )
            if not streaming and duration > 0:
                time.sleep(duration)
        else:
            self.robot.move_cartesian_pose([self.robot.arm_right_name], [commands[0]],
                                           duration=duration, use_wbc=True)
            self.robot.move_joints_position([self.robot.effector_right_name], [commands[1]],
                                            duration=duration, use_wbc=False)

    def step(self, actions: Any) -> dict[str, Any]:
        chunk = np.asarray(actions, np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 10:
            return self._failure(f"Invalid action chunk shape {chunk.shape}")
        executed = 0
        try:
            for target in chunk[: self.config.execute_count]:
                if self.intervened_event.is_set() or self.terminated: break
                current = select_astribot_dims(self._state31(), RIGHT_ARM_GRIPPER)
                self._validate_target(target, current, takeover=False)
                with self.command_lock:
                    if self.intervened_event.is_set(): break
                    self._send_right_target(target, duration=self.config.waypoint_duration, streaming=False)
                if isinstance(self.robot, _FakeRobot): self.robot.state[19:29] = target
                executed += 1
        except Exception as exc:
            self.infrastructure_error = f"{type(exc).__name__}: {exc}"
            return self._failure(self.infrastructure_error, executed=executed)
        try:
            observation = self.observe()
        except Exception as exc:
            return self._failure(f"Observation: {type(exc).__name__}: {exc}", executed=executed)
        response = self._response(observation=observation, executed=executed)
        self.audit.submit({"event": "step", "episode": self._episode_id, "time": time.time(), "model_action": chunk.tolist(),
                           "executed_count": executed, "intervened": response["intervened"],
                           "terminated": response["terminated"], "success": response["success"],
                           "measured_state31": response["observation"]["state"].tolist(),
                           "_images": response["observation"]["images"]})
        return response

    def hold(self) -> dict[str, Any]:
        self.audit.submit({"event": "hold", "episode": self._episode_id, "time": time.time()})
        try: observation = self.observe()
        except Exception as exc: return self._failure(f"Observation: {type(exc).__name__}: {exc}")
        return self._response(observation=observation)

    def _quest_loop(self) -> None:
        interval = 1.0 / max(self.config.quest_poll_rate_hz, 1.0)
        while not self.stop_event.is_set():
            started = time.monotonic()
            with self.quest_lock:
                sample = self.quest.poll()
            self._latest_quest_error = sample.error
            if sample.error:
                self.infrastructure_error = "Quest: " + sample.error
            if sample.outcome:
                self.terminated = True
                self.success = sample.outcome == "success"
                self.audit.submit({"event": "outcome", "episode": self._episode_id,
                                   "time": time.time(), "outcome": sample.outcome})
            if sample.active:
                first_intervention = not self.intervened_event.is_set()
                self.intervened_event.set()
                if first_intervention:
                    self.audit.submit({"event": "intervention", "episode": self._episode_id, "time": time.time()})
                try: self._apply_takeover(sample)
                except Exception as exc: self.infrastructure_error = f"Takeover: {type(exc).__name__}: {exc}"
            else:
                self._takeover_anchor = self._last_takeover_target = None
            self.stop_event.wait(max(0.0, interval - (time.monotonic() - started)))

    def _apply_takeover(self, sample) -> None:
        current = select_astribot_dims(self._state31(), RIGHT_ARM_GRIPPER)
        if self._takeover_anchor is None:
            self._takeover_anchor = current.copy()
            self._last_takeover_target = current.copy()
        target = self._takeover_anchor.copy()
        target[:3] += np.asarray(sample.delta_xyz, np.float32)
        anchor_quat = rotation6d_to_quaternion_xyzw(self._takeover_anchor[3:9])
        target[3:9] = xyzquat_to_xyzrot6d(np.r_[target[:3], quat_multiply_xyzw(anchor_quat, sample.delta_quat_xyzw)])[3:]
        target[9] = sample.gripper
        previous = self._last_takeover_target
        delta = target[:3] - previous[:3]
        distance = float(np.linalg.norm(delta))
        if distance > self.config.max_takeover_translation_step_m:
            target[:3] = previous[:3] + delta / distance * self.config.max_takeover_translation_step_m
        previous_quat = rotation6d_to_quaternion_xyzw(previous[3:9])
        target_quat = rotation6d_to_quaternion_xyzw(target[3:9])
        rotation = self._quat_angle_deg(previous_quat, target_quat)
        if rotation > self.config.max_takeover_rotation_step_deg:
            limited_quat = slerp_xyzw(previous_quat, target_quat,
                                      self.config.max_takeover_rotation_step_deg / rotation)
            target[3:9] = xyzquat_to_xyzrot6d(np.r_[target[:3], limited_quat])[3:]
        self._validate_target(target, previous, takeover=True)
        with self.command_lock: self._send_right_target(target, duration=0.0, streaming=True)
        if isinstance(self.robot, _FakeRobot): self.robot.state[19:29] = target
        self._last_takeover_target = target
        self.audit.submit({"event": "takeover_action", "episode": self._episode_id, "time": time.time(),
                           "target10": target.tolist(), "measured_state31": self._state31().tolist()})

    def _response(self, *, observation=None, executed: int = 0) -> dict[str, Any]:
        invalid = self.intervened_event.is_set() or bool(self.infrastructure_error)
        return {"observation": observation, "executed_count": int(executed), "terminated": bool(self.terminated),
                "success": bool(self.success), "intervened": self.intervened_event.is_set(),
                "invalid_episode": invalid, "error": self.infrastructure_error}

    def _failure(self, error: str, *, executed: int = 0) -> dict[str, Any]:
        self.infrastructure_error = error
        self.audit.submit({"event": "infrastructure_error", "episode": self._episode_id,
                           "time": time.time(), "error": error, "executed_count": int(executed)})
        return self._response(observation=None, executed=executed)

    def close(self) -> None:
        self.stop_event.set()
        self.quest_thread.join(timeout=1)
        self.audit.close()

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "hello": return {"metadata": self.metadata()}
        if operation == "reset": return self.reset(confirm=bool(request.get("confirm", True)))
        if operation == "observe":
            try: return self._response(observation=self.observe())
            except Exception as exc: return self._failure(f"Observation: {type(exc).__name__}: {exc}")
        if operation == "step": return self.step(request.get("actions"))
        if operation == "hold": return self.hold()
        if operation == "close": return {"closed": True}
        return {"error": f"Unknown operation {operation!r}"}


def serve_gateway(gateway: AstribotGateway) -> None:
    from websockets.sync.server import serve
    from .transport import pack, unpack
    def handler(socket):
        for message in socket:
            try: response = gateway.dispatch(unpack(message))
            except Exception as exc: response = {"error": f"{type(exc).__name__}: {exc}"}
            socket.send(pack(response))
    try:
        with serve(handler, gateway.config.host, gateway.config.port, max_size=None, compression=None) as server:
            server.serve_forever()
    finally:
        gateway.close()
