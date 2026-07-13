import json

import numpy as np

from verl.astribot.data.embodiment import (
    DUAL_ARM_GRIPPER,
    RIGHT_ARM_GRIPPER,
    get_astribot_dim,
    select_astribot_dims,
    xyzquat_to_xyzrot6d,
    xyzrot6d_to_xyzquat,
)
from verl.astribot.data.image_composition import preprocess_t_layout
from verl.astribot.data.normalization import FeatureNormalizer
from verl.astribot.protocol import AstribotProtocol


def test_scopes_and_protocol():
    raw = np.arange(31, dtype=np.float32)
    assert get_astribot_dim(RIGHT_ARM_GRIPPER) == 10
    assert get_astribot_dim(DUAL_ARM_GRIPPER) == 20
    np.testing.assert_array_equal(select_astribot_dims(raw, RIGHT_ARM_GRIPPER), raw[19:29])
    np.testing.assert_array_equal(select_astribot_dims(raw, DUAL_ARM_GRIPPER), raw[9:29])
    assert AstribotProtocol(RIGHT_ARM_GRIPPER).action_length == 80
    assert AstribotProtocol(DUAL_ARM_GRIPPER).action_length == 160


def test_rotation_roundtrip():
    xyzquat = np.array([0.1, 0.2, 0.3, 0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    xyzquat[3:] /= np.linalg.norm(xyzquat[3:])
    recovered = xyzrot6d_to_xyzquat(xyzquat_to_xyzrot6d(xyzquat))
    np.testing.assert_allclose(recovered[:3], xyzquat[:3], atol=1e-6)
    assert abs(float(np.dot(recovered[3:], xyzquat[3:]))) > 1 - 1e-5


def test_t_layout_shape():
    images = {
        "base_0_rgb": np.full((256, 320, 3), 10, np.uint8),
        "left_wrist_0_rgb": np.full((128, 160, 3), 20, np.uint8),
        "right_wrist_0_rgb": np.full((128, 160, 3), 30, np.uint8),
    }
    output = preprocess_t_layout(images, (384, 320))
    assert output.shape == (3, 384, 320)
    assert 0 <= output.min() <= output.max() <= 1


def test_normalizer_scope_and_roundtrip(tmp_path):
    path = tmp_path / "norm.json"
    payload = {
        "metadata": {"dimension_scope": RIGHT_ARM_GRIPPER},
        "norm_stats": {"state": {"q01": [0] * 10, "q99": [2] * 10}},
    }
    path.write_text(json.dumps(payload))
    normalizer = FeatureNormalizer(path, expected_scope=RIGHT_ARM_GRIPPER)
    value = np.ones(10, np.float32)
    np.testing.assert_allclose(normalizer.normalize(value, "state"), 0, atol=1e-5)
    np.testing.assert_allclose(normalizer.denormalize(normalizer.normalize(value, "state"), "state"), value, atol=1e-5)
