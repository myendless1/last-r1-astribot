#!/usr/bin/env python3
"""Compute per-dimension q01/q99 statistics for astribot LeRobot datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Layout: left xyz, left quat, left gripper, right xyz, right quat, right gripper.
# Quat dims skip q01/q99 linear normalization (mask=False).
# Gripper dims use q01/q99 (mask=True). In astribot LeRobot data, gripper values
# are "open-ness": larger = more open (1 ≈ open). Left gripper stats often sit
# near ~0.98, meaning usually open during demos—not "closed".
DEFAULT_MASK = (
    [True, True, True, False, False, False, False, True]
    + [True, True, True, False, False, False, False, True]
)


def load_episode_arrays(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    states = np.stack(df["observation.state"].to_numpy())
    actions = np.stack(df["action"].to_numpy())
    return states.astype(np.float64), actions.astype(np.float64)


def compute_quantiles(values: np.ndarray, mask: list[bool], q_low: float = 0.01, q_high: float = 0.99):
    q01 = []
    q99 = []
    for dim in range(values.shape[1]):
        col = values[:, dim]
        if mask[dim]:
            q01.append(float(np.quantile(col, q_low)))
            q99.append(float(np.quantile(col, q_high)))
        else:
            q01.append(float(np.min(col)))
            q99.append(float(np.max(col)))
    return q01, q99


def main():
    parser = argparse.ArgumentParser(description="Compute astribot dataset statistics.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/astribot_filter_300"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/astribot_dataset_statistics.json"),
    )
    parser.add_argument("--dataset-name", type=str, default="astribot_centrifuge_multidrop")
    parser.add_argument("--limit-episodes", type=int, default=None)
    args = parser.parse_args()

    data_dir = args.dataset_root / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    if args.limit_episodes is not None:
        parquet_files = parquet_files[: args.limit_episodes]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    all_states = []
    all_actions = []
    for parquet_path in parquet_files:
        states, actions = load_episode_arrays(parquet_path)
        all_states.append(states)
        all_actions.append(actions)

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    assert states.shape[1] == actions.shape[1] == 16

    mask = DEFAULT_MASK
    state_q01, state_q99 = compute_quantiles(states, mask)
    action_q01, action_q99 = compute_quantiles(actions, mask)

    payload = {
        "__metadata__": {
            "dataset_root": str(args.dataset_root.resolve()),
            "dataset_name": args.dataset_name,
            "num_episodes": len(parquet_files),
            "num_frames": int(len(states)),
        },
        args.dataset_name: {
            "state": {
                "mask": mask,
                "q01": state_q01,
                "q99": state_q99,
            },
            "action": {
                "mask": mask,
                "q01": action_q01,
                "q99": action_q99,
            },
        }
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote statistics for {len(parquet_files)} episodes / {len(states)} frames -> {args.output}")


if __name__ == "__main__":
    main()
