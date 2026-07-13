#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from verl.astribot.data.embodiment import ACTION_KEY, IMAGE_KEYS, RAW_DIM, STATE_KEY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    info_path = args.root / "meta" / "info.json"
    if not info_path.exists(): raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    files = sorted((args.root / "data").glob("**/*.parquet"))
    if not files: raise FileNotFoundError("No LeRobot parquet files")
    frames = 0; episodes = set()
    required = {"episode_index", STATE_KEY, ACTION_KEY, *IMAGE_KEYS}
    for path in files:
        table = pq.read_table(path)
        missing = required - set(table.column_names)
        if missing: raise KeyError(f"{path}: missing {sorted(missing)}")
        states, actions = np.stack(table.column(STATE_KEY).to_pylist()), np.stack(table.column(ACTION_KEY).to_pylist())
        if states.shape[1:] != (RAW_DIM,) or actions.shape[1:] != (RAW_DIM,): raise ValueError(f"{path}: invalid vector shapes")
        if not np.isfinite(states).all() or not np.isfinite(actions).all(): raise ValueError(f"{path}: non-finite robot values")
        frames += len(table); episodes.update(table.column("episode_index").to_pylist())
    print(json.dumps({"codebase_version": info.get("codebase_version"), "fps": info.get("fps"), "frames": frames, "episodes": len(episodes)}, indent=2))


if __name__ == "__main__": main()
