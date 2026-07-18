#!/usr/bin/env python3
"""Diagnose command/state lag and spatial residuals in the Astribot dataset."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ARMS = {"left": (0, 3), "right": (8, 11)}


def qangle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = q1 / np.maximum(np.linalg.norm(q1, axis=1, keepdims=True), 1e-12)
    q2 = q2 / np.maximum(np.linalg.norm(q2, axis=1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(q1 * q2, axis=1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def qlog_error(q_target: np.ndarray, q_actual: np.ndarray) -> np.ndarray:
    """Rotation vector taking target orientation to actual orientation, wxyz."""
    a = q_target / np.maximum(np.linalg.norm(q_target, axis=1, keepdims=True), 1e-12)
    b = q_actual / np.maximum(np.linalg.norm(q_actual, axis=1, keepdims=True), 1e-12)
    aw, av = a[:, :1], a[:, 1:]
    bw, bv = b[:, :1], b[:, 1:]
    # conj(a) * b
    w = aw * bw + np.sum(av * bv, axis=1, keepdims=True)
    v = aw * bv - bw * av - np.cross(av, bv)
    flip = w[:, 0] < 0
    w[flip] *= -1
    v[flip] *= -1
    vn = np.linalg.norm(v, axis=1)
    ang = 2.0 * np.arctan2(vn, np.clip(w[:, 0], -1.0, 1.0))
    return v * (ang / np.maximum(vn, 1e-12))[:, None]


def aligned(a: np.ndarray, s: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Positive lag means state realizes the earlier command after `lag` frames."""
    if lag >= 0:
        return a[: len(a) - lag or None], s[lag:]
    return a[-lag:], s[: len(s) + lag]


def load_episode(root: Path, ep: int):
    p = root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    table = pq.read_table(p, columns=["observation.state", "action"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    meta = json.loads((root / "source_meta" / f"episode_{ep:06d}.json").read_text())
    source = meta["source_meta"]["source_path"]
    task = "centrifuge" if "centrifuge" in source else "multidrop"
    return task, action, state


def quantiles(x: np.ndarray):
    return {str(q): float(v) for q, v in zip((0.5, 0.9, 0.95, 0.99), np.quantile(x, (0.5, 0.9, 0.95, 0.99)))}


def lag_scan(episodes, arm: str, max_lag: int):
    p0, q0 = ARMS[arm]
    accum = {k: defaultdict(float) for k in range(-max_lag, max_lag + 1)}
    per_episode = []
    for ep, _, action, state in episodes:
        local = {}
        for k in range(-max_lag, max_lag + 1):
            a, s = aligned(action, state, k)
            dp = s[:, p0:p0+3] - a[:, p0:p0+3]
            da = np.diff(a[:, p0:p0+3], axis=0)
            ds = np.diff(s[:, p0:p0+3], axis=0)
            # A centered derivative loss is insensitive to static Cartesian bias.
            dv = ds - da
            n = len(dp)
            accum[k]["n"] += n
            accum[k]["pos_sse"] += np.sum(dp * dp)
            accum[k]["vel_sse"] += np.sum(dv * dv)
            accum[k]["vel_energy"] += np.sum(da * da) + 1e-15
            accum[k]["ang_sq"] += np.sum(qangle(a[:, q0:q0+4], s[:, q0:q0+4]) ** 2)
            local[k] = np.mean(dv * dv)
        per_episode.append(min(local, key=local.get))
    rows = {}
    for k, d in accum.items():
        rows[k] = {
            "position_rmse_m": float(np.sqrt(d["pos_sse"] / d["n"])),
            "orientation_rmse_deg": float(np.degrees(np.sqrt(d["ang_sq"] / d["n"]))),
            "derivative_relative_rmse": float(np.sqrt(d["vel_sse"] / d["vel_energy"])),
        }
    best_dynamic = min(rows, key=lambda k: rows[k]["derivative_relative_rmse"])
    best_position = min(rows, key=lambda k: rows[k]["position_rmse_m"])
    return rows, int(best_dynamic), int(best_position), per_episode


def fractional_lag_scan(episodes, arm: str, lo: float = -0.5, hi: float = 2.5, step: float = 0.05):
    """Fine scan on the uniform processed timeline using linear/nlerp interpolation."""
    p0, q0 = ARMS[arm]
    lags = np.round(np.arange(lo, hi + step / 2, step), 6)
    acc = {float(k): defaultdict(float) for k in lags}
    for _, _, action, state in episodes:
        n = len(action)
        for lag in lags:
            start = max(0, int(np.ceil(-lag)))
            stop = min(n, int(np.ceil(n - 1 - lag)))
            idx = np.arange(start, stop)
            u = idx + lag
            j = np.floor(u).astype(int)
            w = (u - j)[:, None]
            sp = state[j, p0:p0+3] * (1-w) + state[j+1, p0:p0+3] * w
            q1 = state[j, q0:q0+4]
            q2 = state[j+1, q0:q0+4].copy()
            q2[np.sum(q1*q2, axis=1) < 0] *= -1
            sq = q1 * (1-w) + q2 * w
            a = action[idx]
            dp = sp - a[:, p0:p0+3]
            dv = np.diff(sp, axis=0) - np.diff(a[:, p0:p0+3], axis=0)
            d = acc[float(lag)]
            d["n"] += len(dp); d["pos_sse"] += np.sum(dp*dp)
            d["vel_sse"] += np.sum(dv*dv); d["vel_energy"] += np.sum(np.diff(a[:, p0:p0+3],axis=0)**2)+1e-15
            d["ang_sq"] += np.sum(qangle(a[:, q0:q0+4], sq)**2)
    rows={k:{"position_rmse_m":float(np.sqrt(d["pos_sse"]/d["n"])),
             "orientation_rmse_deg":float(np.degrees(np.sqrt(d["ang_sq"]/d["n"]))),
             "derivative_relative_rmse":float(np.sqrt(d["vel_sse"]/d["vel_energy"]))} for k,d in acc.items()}
    return rows


def spatial_bin_predictability(pos, residual, episode_ids, bins=5):
    train = episode_ids % 5 != 0
    test = ~train
    edges = [np.unique(np.quantile(pos[train, j], np.linspace(0, 1, bins + 1))) for j in range(3)]
    if any(len(e) < 3 for e in edges):
        return {"cv_r2": float("nan"), "between_bin_fraction": float("nan"), "populated_bins": 0}

    def ids(x):
        ix = [np.clip(np.searchsorted(edges[j][1:-1], x[:, j]), 0, bins - 1) for j in range(3)]
        return ix[0] * bins * bins + ix[1] * bins + ix[2]

    bi = ids(pos)
    global_mean = residual[train].mean(axis=0)
    means = {}
    counts = {}
    for b in np.unique(bi[train]):
        m = train & (bi == b)
        if m.sum() >= 100:
            means[int(b)] = residual[m].mean(axis=0)
            counts[int(b)] = int(m.sum())
    pred = np.repeat(global_mean[None], test.sum(), axis=0)
    test_bins = bi[test]
    for b, mean in means.items():
        pred[test_bins == b] = mean
    y = residual[test]
    sse = np.sum((y - pred) ** 2)
    sst = np.sum((y - global_mean) ** 2)
    cv_r2 = 1.0 - sse / max(sst, 1e-15)
    weights = np.array(list(counts.values()), dtype=float)
    mm = np.array([means[b] for b in counts])
    between = np.sum(weights[:, None] * (mm - global_mean) ** 2)
    total = np.sum((residual[train] - global_mean) ** 2)
    ranges = np.ptp(mm, axis=0) if len(mm) else np.full(3, np.nan)
    return {
        "cv_r2": float(cv_r2),
        "between_bin_fraction": float(between / max(total, 1e-15)),
        "populated_bins": len(means),
        "bin_mean_component_range_mm": (1000 * ranges).tolist(),
    }


def residual_stats(episodes, arm: str, lag: int, fps: float):
    p0, q0 = ARMS[arm]
    all_pos, all_dp, all_dr, all_speed, all_ep = [], [], [], [], []
    for ep, _, action, state in episodes:
        a, s = aligned(action, state, lag)
        pos = a[:, p0:p0+3]
        dp = s[:, p0:p0+3] - pos
        dr = qlog_error(a[:, q0:q0+4], s[:, q0:q0+4])
        speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps]
        all_pos.append(pos); all_dp.append(dp); all_dr.append(dr); all_speed.append(speed)
        all_ep.append(np.full(len(pos), ep, dtype=np.int32))
    pos = np.concatenate(all_pos); dp = np.concatenate(all_dp); dr = np.concatenate(all_dr)
    speed = np.concatenate(all_speed); epids = np.concatenate(all_ep)
    pn = np.linalg.norm(dp, axis=1); rn = np.linalg.norm(dr, axis=1)
    slow = speed < 0.01
    result = {
        "n": len(dp),
        "position_bias_mm": (1000 * dp.mean(axis=0)).tolist(),
        "position_error_mm": {k: 1000*v for k, v in quantiles(pn).items()},
        "position_rmse_mm": float(1000 * np.sqrt(np.mean(pn**2))),
        "position_centered_rmse_mm": float(1000 * np.sqrt(np.mean((dp-dp.mean(axis=0))**2))),
        "orientation_bias_deg_rotvec": np.degrees(dr.mean(axis=0)).tolist(),
        "orientation_error_deg": {k: float(np.degrees(v)) for k, v in quantiles(rn).items()},
        "slow_fraction": float(slow.mean()),
        "slow_position_bias_mm": (1000 * dp[slow].mean(axis=0)).tolist(),
        "slow_position_error_mm": {k: 1000*v for k, v in quantiles(np.linalg.norm(dp[slow],axis=1)).items()},
        "slow_orientation_error_deg": {k: float(np.degrees(v)) for k, v in quantiles(np.linalg.norm(dr[slow],axis=1)).items()},
        "spatial_all": spatial_bin_predictability(pos, dp, epids),
        "spatial_slow": spatial_bin_predictability(pos[slow], dp[slow], epids[slow]),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--max-lag", type=int, default=12)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    info = json.loads((args.root / "meta/info.json").read_text())
    episodes = []
    for ep in range(info["total_episodes"]):
        task, action, state = load_episode(args.root, ep)
        episodes.append((ep, task, action, state))

    out = {"fps": info["fps"], "episodes": len(episodes), "frames": sum(len(x[2]) for x in episodes), "groups": {}}
    groups = {"all": episodes}
    groups.update({task: [x for x in episodes if x[1] == task] for task in ("centrifuge", "multidrop")})
    for group, eps in groups.items():
        out["groups"][group] = {}
        for arm in ARMS:
            scan, best_dynamic, best_position, per_ep = lag_scan(eps, arm, args.max_lag)
            fine = fractional_lag_scan(eps, arm)
            fine_best_dynamic = min(fine, key=lambda k: fine[k]["derivative_relative_rmse"])
            fine_best_position = min(fine, key=lambda k: fine[k]["position_rmse_m"])
            fine_best_orientation = min(fine, key=lambda k: fine[k]["orientation_rmse_deg"])
            out["groups"][group][arm] = {
                "episode_count": len(eps),
                "lag_scan": {str(k): v for k, v in scan.items()},
                "best_dynamic_lag_frames": best_dynamic,
                "best_position_lag_frames": best_position,
                "fractional_best_dynamic_lag_frames": fine_best_dynamic,
                "fractional_best_position_lag_frames": fine_best_position,
                "fractional_best_orientation_lag_frames": fine_best_orientation,
                "fractional_best_metrics": {
                    "dynamic": fine[fine_best_dynamic],
                    "position": fine[fine_best_position],
                    "orientation": fine[fine_best_orientation],
                },
                "per_episode_dynamic_lag_quantiles_frames": quantiles(np.asarray(per_ep)),
                "per_episode_dynamic_lag_mode_frames": int(np.bincount(np.asarray(per_ep)+args.max_lag).argmax()-args.max_lag),
                "residual_at_dynamic_lag": residual_stats(eps, arm, best_dynamic, info["fps"]),
                "residual_at_zero_lag": residual_stats(eps, arm, 0, info["fps"]),
            }
    text = json.dumps(out, indent=2, allow_nan=True)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
