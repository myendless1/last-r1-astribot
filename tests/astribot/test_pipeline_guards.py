from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from verl.astribot.data.collate import astribot_sft_collate
from verl.astribot.data.latent_cache import LatentCache
from verl.astribot.deploy.robot_io import read_state31
from verl.astribot.deploy.safety import validate_action_chunk


def _sample(action_dim: int = 10) -> dict:
    return {
        "input_ids_prompt": torch.tensor([10, 11, 12]),
        "pixel_values": torch.zeros(4, 8),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "latent_gt_embeds": torch.zeros(2, 8),
        "latent_mask": torch.ones(2),
        "action_token_ids": torch.arange(2 * action_dim).reshape(2, action_dim) + 100,
        "action_loss_mask": torch.ones(2, action_dim),
    }


def test_collate_does_not_expose_action_labels():
    batch = astribot_sft_collate([_sample()], pad_token_id=0, latent_pad_id=7, latent_end_id=8, max_prompt_length=5)
    assert torch.all(batch["input_ids"][:, -20:] == 7)
    assert torch.any(batch["action_labels"] != 7)


def test_latent_cache_is_bounded(tmp_path):
    metadata = {"latent_length": 2, "hidden_size": 3}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    for episode in range(3):
        np.savez(tmp_path / f"episode_{episode:06d}.npz", latents=np.zeros((1, 2, 3)), mask=np.ones((1, 2)))
    cache = LatentCache(tmp_path, latent_length=2, hidden_size=3, max_cached_episodes=2)
    for episode in range(3):
        cache.get(episode, 0)
    assert list(cache._episodes) == [1, 2]


class _FakeRobot:
    torso_name = "torso"
    arm_left_name = "left"
    arm_right_name = "right"
    effector_left_name = "left_gripper"
    effector_right_name = "right_gripper"
    head_name = "head"

    def get_current_cartesian_pose(self, names):
        del names
        identity = np.array([0, 0, 0.5, 0, 0, 0, 1], np.float32)
        return [identity, identity, identity]

    def get_current_joints_position(self, names):
        del names
        return [np.array([12.0]), np.array([34.0]), np.array([0.1, -0.1])]


def test_read_state31_returns_complete_state():
    state = read_state31(_FakeRobot())
    assert state.shape == (31,)
    assert state[18] == 12.0
    assert state[28] == 34.0


def test_safety_checks_first_step_against_current():
    current = np.zeros(10, np.float32)
    actions = np.zeros((2, 10), np.float32)
    actions[:, 0] = 0.2
    with pytest.raises(ValueError, match="First action"):
        validate_action_chunk(actions, current=current)
