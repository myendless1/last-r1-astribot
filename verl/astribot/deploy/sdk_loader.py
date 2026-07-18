from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_ASTRIBOT_SDK_ROOT = Path("/opt/astribot_sdk")


def load_astribot_class(sdk_root: str | os.PathLike[str] | None = None) -> Any:
    root = Path(sdk_root or os.environ.get("ASTRIBOT_SDK_ROOT", DEFAULT_ASTRIBOT_SDK_ROOT)).expanduser().resolve()
    if not (root / "core").is_dir():
        raise RuntimeError(f"Astribot SDK path does not look valid: {root}")
    for path in (root, root / "core" / "common"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    os.environ.setdefault("ASTRIBOT_SDK_ROOT", str(root))
    os.environ.setdefault("ROBOT_TYPE", "S1")
    try:
        from core.astribot_api.astribot_client import Astribot
    except Exception as exc:
        raise RuntimeError(
            "Unable to import Astribot SDK. Source ROS Noetic and the SDK env.sh before starting the gateway."
        ) from exc
    return Astribot
