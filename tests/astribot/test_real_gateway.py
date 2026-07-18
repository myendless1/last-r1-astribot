import threading

import numpy as np
import pytest

from verl.astribot.data.embodiment import RIGHT_ARM_GRIPPER, select_astribot_dims
from verl.astribot.deploy.gateway import AstribotGateway, GatewayConfig
from verl.astribot.deploy.quest import (
    QuestController,
    QuestSample,
    index_trigger_to_gripper,
    webxr_delta_to_robot,
)


def _target(gateway):
    return select_astribot_dims(gateway._state31(), RIGHT_ARM_GRIPPER).copy()


def test_webxr_axis_and_gripper_mapping():
    np.testing.assert_allclose(webxr_delta_to_robot([1, 2, 3]), [-3, -1, 2])
    assert index_trigger_to_gripper(0.0, 0.2) == pytest.approx(100.0)
    assert index_trigger_to_gripper(1.0, 0.2) == pytest.approx(0.0)


def test_quest_buttons_require_release_then_emit_edges():
    snapshots = iter([
        {"hands": {"right": {"valid": True, "buttons": [0, 0, 0, 0, 1, 0], "middle": 0}}},
        {"hands": {"right": {"valid": True, "buttons": [0] * 6, "middle": 0}}},
        {"hands": {"right": {"valid": True, "buttons": [0, 0, 0, 0, 1, 0], "middle": 0}}},
    ])
    quest = QuestController("", snapshot_provider=lambda: next(snapshots))
    assert quest.poll().outcome is None
    assert quest.poll().outcome is None
    assert quest.poll().outcome == "success"


def test_fake_gateway_executes_safe_right_action(tmp_path):
    gateway = AstribotGateway(GatewayConfig(fake=True, quest_state_url="", log_dir=str(tmp_path)))
    try:
        target = _target(gateway)
        target[0] += 0.01
        result = gateway.step(target[None])
        assert result["executed_count"] == 1
        assert not result["invalid_episode"]
        np.testing.assert_allclose(_target(gateway), target)
    finally:
        gateway.close()


def test_unsafe_policy_action_invalidates_episode(tmp_path):
    gateway = AstribotGateway(GatewayConfig(fake=True, quest_state_url="", log_dir=str(tmp_path)))
    try:
        target = _target(gateway)
        target[0] += 0.2
        result = gateway.step(target[None])
        assert result["executed_count"] == 0
        assert result["invalid_episode"]
        assert "Translation step" in result["error"]
    finally:
        gateway.close()


def test_takeover_marks_episode_invalid_and_moves_from_anchor(tmp_path):
    gateway = AstribotGateway(GatewayConfig(fake=True, quest_state_url="", robot_command_enabled=True,
                                            log_dir=str(tmp_path)))
    try:
        gateway.intervened_event.set()
        gateway._apply_takeover(QuestSample(active=True, delta_xyz=np.asarray([0.005, 0, 0]),
                                             delta_quat_xyzw=np.asarray([0, 0, 0, 1]), gripper=50))
        assert gateway.hold()["invalid_episode"]
        assert _target(gateway)[0] == pytest.approx(0.405)
        assert _target(gateway)[9] == pytest.approx(50)
    finally:
        gateway.close()


def test_gateway_serializes_policy_and_takeover_commands(tmp_path):
    gateway = AstribotGateway(GatewayConfig(fake=True, quest_state_url="", log_dir=str(tmp_path)))
    assert isinstance(gateway.command_lock, type(threading.RLock()))
    gateway.close()
