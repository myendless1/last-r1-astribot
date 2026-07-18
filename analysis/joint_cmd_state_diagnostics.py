#!/usr/bin/env python3
"""Scan command/state lag in raw Astribot arm joint trajectories."""
from __future__ import annotations

import argparse
import glob
import json

import h5py
import numpy as np


def aligned(a, s, lag):
    if lag >= 0:
        return a[: len(a)-lag or None], s[lag:]
    return a[-lag:], s[:len(s)+lag]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    arms = {"left": slice(7, 14), "right": slice(15, 22)}
    out = {}
    for root in args.roots:
        task = root.rstrip("/").split("/")[-1]
        acc = {arm: {k: [0.0, 0.0, 0.0, 0] for k in range(-6, 7)} for arm in arms}
        for path in glob.glob(root.rstrip("/") + "/*.hdf5"):
            with h5py.File(path) as f:
                qc = np.asarray(f["joints_dict/joints_position_command"], dtype=float)
                qs = np.asarray(f["joints_dict/joints_position_state"], dtype=float)
                tc = np.asarray(f["joints_dict/command_timestamp"], dtype=float)
                ts = np.asarray(f["joints_dict/state_timestamp"], dtype=float)
                lo, hi = max(tc[0], ts[0]), min(tc[-1], ts[-1])
                t = np.arange(lo, hi, 1/30)
                if len(t) < 10:
                    continue
                for arm, sl in arms.items():
                    c = np.stack([np.interp(t, tc, qc[:, j]) for j in range(sl.start, sl.stop)], axis=1)
                    s = np.stack([np.interp(t, ts, qs[:, j]) for j in range(sl.start, sl.stop)], axis=1)
                    for k in range(-6, 7):
                        a, b = aligned(c, s, k)
                        e = b-a; de=np.diff(b,axis=0)-np.diff(a,axis=0)
                        z=acc[arm][k]; z[0]+=np.sum(e*e); z[1]+=np.sum(de*de); z[2]+=np.sum(np.diff(a,axis=0)**2); z[3]+=e.size
        out[task]={}
        for arm, scan in acc.items():
            rows={k:{"joint_rmse_deg":float(np.degrees(np.sqrt(v[0]/v[3]))),
                     "derivative_relative_rmse":float(np.sqrt(v[1]/max(v[2],1e-15)))} for k,v in scan.items()}
            out[task][arm]={"scan":rows,"best_lag_frames":min(rows,key=lambda k:rows[k]["derivative_relative_rmse"])}
    with open(args.output,"w") as f:
        json.dump(out,f,indent=2)


if __name__ == "__main__":
    main()
