#!/usr/bin/env python3
"""Convert Astribot teleoperation HDF5 episodes to LeRobot 0.4.4."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from verl.astribot.data.embodiment import ACTION_KEY, IMAGE_KEYS, STATE_KEY, build_cartesian31


CAMERAS = {"head": IMAGE_KEYS[0], "left": IMAGE_KEYS[1], "right": IMAGE_KEYS[2]}


def _decode_cache(handle: h5py.File):
    result = {}
    for name in CAMERAS:
        sizes = np.asarray(handle[f"images_dict/{name}/rgb_size"], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(sizes[:-1])]
        result[name] = (handle[f"images_dict/{name}/rgb"], sizes, offsets)
    return result


def _image(cache, name: str, index: int):
    data, sizes, offsets = cache[name]
    encoded = np.asarray(data[int(offsets[index]) : int(offsets[index] + sizes[index])], dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError(f"Could not decode {name} frame {index}")
    return np.transpose(bgr[..., ::-1], (2, 0, 1)).copy()


def _first(handle, keys):
    for key in keys:
        if key in handle: return np.asarray(handle[key])
    raise KeyError(f"Missing all HDF5 keys: {keys}")


def _poses(handle, command: bool):
    prefix = "command_poses_dict" if command else "poses_dict"
    fallback = "command_poses_dict"
    return {
        name: _first(handle, [f"{prefix}/astribot_{robot}", f"{fallback}/astribot_{robot}"])
        for name, robot in (("torso", "torso"), ("left", "arm_left"), ("right", "arm_right"))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", default="S1-stationary")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    episodes = sorted(args.raw_path.glob("*.hdf5"))
    if not episodes: raise FileNotFoundError(f"No HDF5 episodes under {args.raw_path}")
    if args.output_path.exists():
        if not args.overwrite: raise FileExistsError(args.output_path)
        shutil.rmtree(args.output_path)
    with h5py.File(episodes[0]) as handle:
        cache = _decode_cache(handle)
        samples = {target: _image(cache, source, 0) for source, target in CAMERAS.items()}
    features = {key: {"dtype": "image", "shape": value.shape, "names": ["channel", "height", "width"]} for key, value in samples.items()}
    features.update({STATE_KEY: {"dtype": "float32", "shape": (31,), "names": [f"state_{i}" for i in range(31)]},
                     ACTION_KEY: {"dtype": "float32", "shape": (31,), "names": [f"action_{i}" for i in range(31)]}})
    dataset = LeRobotDataset.create(repo_id=args.output_path.name, root=args.output_path, fps=args.fps,
        robot_type=args.robot_type, features=features, use_videos=False)
    for episode in episodes:
        with h5py.File(episode) as handle:
            cache = _decode_cache(handle)
            command_poses, state_poses = _poses(handle, True), _poses(handle, False)
            command_joints = np.asarray(handle["joints_dict/joints_position_command"])
            state_joints = _first(handle, ["joints_dict/joints_position_state", "joints_dict/joints_position_command"])
            for index in range(len(command_joints)):
                frame = {"task": args.task, STATE_KEY: build_cartesian31(state_poses, state_joints, index),
                         ACTION_KEY: build_cartesian31(command_poses, command_joints, index)}
                frame.update({target: _image(cache, source, index) for source, target in CAMERAS.items()})
                dataset.add_frame(frame)
            dataset.save_episode()


if __name__ == "__main__": main()
