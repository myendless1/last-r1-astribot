from __future__ import annotations

import h5py
import numpy as np

from ..data.embodiment import build_cartesian31, select_astribot_dims
from .action_adapter import action_to_commands
from .safety import validate_action_chunk


def create_robot(high_control_rights: bool = True):
    from core.astribot_api.astribot_client import Astribot

    robot = Astribot(high_control_rights=high_control_rights)
    robot.set_head_follow_effector(False)
    return robot


def read_images(robot) -> dict[str, np.ndarray]:
    rgb, _, _, _ = robot.get_images_dict()
    aliases = {
        "base_0_rgb": "Bolt",
        "left_wrist_0_rgb": "left_D405",
        "right_wrist_0_rgb": "right_D405",
    }
    missing = [source for source in aliases.values() if source not in rgb]
    if missing:
        raise TimeoutError(f"Missing Astribot cameras: {missing}")
    return {target: rgb[source] for target, source in aliases.items()}


def read_state31(robot) -> np.ndarray:
    poses = robot.get_current_cartesian_pose(names=[robot.torso_name, robot.arm_left_name, robot.arm_right_name])
    joints = robot.get_current_joints_position(
        names=[robot.effector_left_name, robot.effector_right_name, robot.head_name]
    )
    # build_cartesian31 consumes the grippers and head from this recorded 22D layout.
    tail = np.zeros(22, np.float32)
    tail[11] = np.asarray(joints[0]).reshape(-1)[0]
    tail[19] = np.asarray(joints[1]).reshape(-1)[0]
    tail[20:22] = np.asarray(joints[2]).reshape(-1)[:2]
    return build_cartesian31(
        {"torso": np.asarray([poses[0]]), "left": np.asarray([poses[1]]), "right": np.asarray([poses[2]])},
        np.asarray([tail]),
        0,
    )


def move_to_initial_pose(robot, hdf5_path: str, *, frame_index: int = 0, duration: float = 5.0) -> None:
    """Move to a recorded Astribot joint command before Cartesian policy control."""
    with h5py.File(hdf5_path, "r") as episode:
        joints = episode.get("joints_dict/joints_position_command")
        if joints is None:
            joints = episode.get("joints_dict/joints_position")
        if joints is None:
            raise KeyError("HDF5 has neither joints_position_command nor joints_position")
        tail = np.asarray(joints[frame_index], dtype=np.float32).reshape(-1)[-22:]
    commands = [tail[0:4], tail[4:11], tail[11:12], tail[12:19], tail[19:20], tail[20:22]]
    robot.move_joints_position(robot.whole_body_names[1:], commands, duration=duration, use_wbc=False)


def execute_chunk(
    robot,
    actions: np.ndarray,
    *,
    scope: str,
    execute_count: int = 1,
    duration: float = 0.3,
) -> None:
    current = select_astribot_dims(read_state31(robot), scope)
    validate_action_chunk(actions, current=current)
    names = robot.whole_body_names[1:6]
    for action in actions[:execute_count]:
        commands = action_to_commands(action, scope)
        command_names = [names[3], names[4]] if len(commands) == 2 else [names[1], names[2], names[3], names[4]]
        robot.move_cartesian_pose(command_names, commands, duration=duration, use_wbc=True)
