#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from verl.astribot.data.embodiment import (
    ACTION_KEY,
    STATE_KEY,
    get_astribot_dim,
    normalize_dimension_scope,
    select_astribot_dims,
)
from verl.astribot.data.normalization import RunningStats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension-scope", required=True, choices=("right_arm_gripper", "dual_arm_gripper"))
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--action-frame-stride", type=int, default=1)
    args = parser.parse_args()
    if args.action_horizon <= 0 or args.action_frame_stride <= 0:
        raise ValueError("action-horizon and action-frame-stride must be positive")

    scope = normalize_dimension_scope(args.dimension_scope)
    state_stats, action_stats = RunningStats(), RunningStats()
    conditions = 0
    episodes_used = 0
    for path in sorted((args.root / "data").glob("**/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", STATE_KEY, ACTION_KEY], memory_map=True)
        episodes = np.asarray(table.column("episode_index").to_numpy(), np.int64)
        states = select_astribot_dims(np.stack(table.column(STATE_KEY).to_pylist()), scope)
        actions = select_astribot_dims(np.stack(table.column(ACTION_KEY).to_pylist()), scope)
        cuts = np.flatnonzero(np.diff(episodes)) + 1
        for start, end in zip(np.r_[0, cuts], np.r_[cuts, len(episodes)]):
            length = int(end - start)
            if length <= 0:
                continue
            state_stats.update(states[start:end])
            repeats = np.zeros(length, dtype=np.int64)
            for condition in range(length):
                future = condition + np.arange(args.action_horizon) * args.action_frame_stride
                repeats[future[future < length]] += 1
            action_stats.update(np.repeat(actions[start:end], repeats, axis=0))
            conditions += length
            episodes_used += 1

    if not conditions:
        raise ValueError("No Astribot frames found")
    dim = get_astribot_dim(scope)
    payload = {
        "metadata": {
            "dimension_scope": scope,
            "state_dim": dim,
            "action_dim": dim,
            "action_horizon": args.action_horizon,
            "action_frame_stride": args.action_frame_stride,
            "dataset_root": str(args.root.resolve()),
            "conditions": conditions,
            "episodes": episodes_used,
        },
        "norm_stats": {STATE_KEY: state_stats.result(), ACTION_KEY: action_stats.result()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
