from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .data.embodiment import get_astribot_dim, normalize_dimension_scope


PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class AstribotProtocol:
    dimension_scope: str
    action_horizon: int = 8
    latent_length: int = 8
    image_height: int = 384
    image_width: int = 320
    max_prompt_length: int = 576
    action_frame_stride: int = 1
    future_frame_stride: int = 1
    image_layout: str = "t_layout"
    image_color_order: str = "rgb"
    lerobot_version: str = "0.4.4"
    action_bins: int = 256
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension_scope", normalize_dimension_scope(self.dimension_scope))
        if self.action_horizon <= 0 or self.latent_length <= 0 or self.max_prompt_length <= 0 or self.action_frame_stride <= 0 or self.future_frame_stride <= 0:
            raise ValueError("action_horizon, latent_length, max_prompt_length, and frame strides must be positive")
        if self.image_layout != "t_layout" or self.image_color_order != "rgb":
            raise ValueError("Astribot protocol requires RGB T-layout images")

    @property
    def state_dim(self) -> int:
        return get_astribot_dim(self.dimension_scope)

    @property
    def action_dim(self) -> int:
        return self.state_dim

    @property
    def action_length(self) -> int:
        return self.action_horizon * self.action_dim

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "state_dim": self.state_dim, "action_dim": self.action_dim}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def assert_compatible(self, other: Mapping[str, Any]) -> None:
        expected = self.to_dict()
        keys = ("protocol_version", "dimension_scope", "state_dim", "action_dim", "action_horizon", "latent_length", "image_layout", "max_prompt_length", "action_frame_stride", "future_frame_stride")
        mismatches = [f"{key}: expected {expected[key]!r}, got {other.get(key)!r}" for key in keys if other.get(key) != expected[key]]
        if mismatches:
            raise ValueError("Incompatible Astribot protocol: " + "; ".join(mismatches))


def load_protocol(path: str | Path) -> AstribotProtocol:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = AstribotProtocol.__dataclass_fields__.keys()
    return AstribotProtocol(**{key: value for key, value in payload.items() if key in allowed})
