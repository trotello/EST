# scripts/eval_bc.py
# -*- coding: utf-8 -*-
"""
Compute offline MSE of predicted actions vs ground-truth actions.

Metrics:
  - overall MSE
  - MSE near boundaries (±window frames around boundary_hard==1)
  - MSE inside events (frames farther than window from any boundary)

Usage:
  python -m scripts.eval_bc \
    --work-dir ./work \
    --model-key clip_clip-vit-base-patch16 \
    --variant est \
    --dataset-root /PATH/TO/LIBERO/data \
    --split val \
    --window-sec 0.5 \
    --fps 12
"""

from __future__ import annotations
import argparse, os, glob
import numpy as np

from estlib.data.libero_actions import read_actions_from_h5

def _episode_rel_path_from_base(base: str) -> str:
    # reverse our "flatten" (ep_<relpath with __ separators>)
    rel = base.replace("ep_", "")
    return rel.replace("__", os.sep)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--variant", choices=["est","plain"], required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--window-sec", type=float, default=0.5)
    p.add_argument("--fps", type=float, default=12.0)
    args = p.parse_args()

    preds_dir = os.path.join(args.work_dir, "preds", f"{args.variant}_{args.model_key}")
    feat_dir  = os.path.join(args.work_dir, "features", args.model_key)
    ev_dir    = os.path.join(args.work_dir, "events",   args.model_key)
    split_dir = os.path.join(args.dataset_root, args.split)

    pred_files = sorted(glob.glob(os.path.join(preds_dir, "ep_*_ahat.npy")))
    if not pred_files:
        raise FileNotFoundError(f"No predictions found in {preds_dir}. Run predict_actions first.")

    win = int(round(args.window_sec * args.fps)) if hasattr(args, "window_sec") else int(round(args.window_sec * args.fps))
    # argparse hyphen bug workaround:
    win = int(round(args.window_sec * args.fps))

    overall_num, overall_den = 0.0, 0
    near_num, near_den = 0.0, 0
    inner_num, inner_den = 0.0, 0

    per_ep = []

    for pf in pred_files:
        base = os.path.basename(pf).replace("_ahat.npy", "")
        a_hat = np.load(pf).astype("float32")
        t = np.load(os.path.join(preds_dir, base + "_t.npy")).astype("float32")
        T = len(a_hat)

        # ground-truth actions (align via nearest timestamp)
        rel = _episode_rel_path_from_base(base)  # e.g., task/episode.h5 or task/video.mp4
        # prefer HDF5 beside that path
        stem = os.path.splitext(rel)[0]
        h5_path = None
        for ext in (".h5", ".hdf5"):
            cand = os.path.join(split_dir, stem + ext)
            if os.path.isfile(cand):
                h5_path = cand; break
        if h5_path is None:
            # skip if we can't evaluate (e.g., no actions for this episode)
            continue

        acts, t_a = read_actions_from_h5(h5_path)
        # align frame times to nearest action index
        idx = np.searchsorted(t_a, t, side="left")
        idx = np.clip(idx, 0, len(t_a)-1)
        a = acts[idx].astype("float32")
        A = a.shape[1]
        T = min(T, len(a))
        a = a[:T]; a_hat = a_hat[:T]

        # events for boundary masks
        ev_path = os.path.join(ev_dir, base + ".npz")
        if not os.path.isfile(ev_path):
            # build masks as all-inner if no sidecar (rare)
            near_mask = np.zeros((T,), dtype=bool)
        else:
            ev = np.load(ev_path)
            bh = ev["boundary_hard"].astype("int32")
            # build a mask that's True within ±win frames of any boundary
            near_mask = np.zeros((T,), dtype=bool)
            where = np.where(bh[:T] > 0)[0]
            for t0 in where:
                s = max(0, t0 - win); e = min(T, t0 + win + 1)
                near_mask[s:e] = True

        inner_mask = ~near_mask

        # MSEs
        err = (a_hat - a) ** 2
        mse_all = float(err.mean())
        if near_mask.any():
            mse_near = float(err[near_mask].mean())
        else:
            mse_near = float("nan")
        if inner_mask.any():
            mse_inner = float(err[inner_mask].mean())
        else:
            mse_inner = float("nan")

        per_ep.append((base, mse_all, mse_near, mse_inner))

        # accumulate
        overall_num += float(err.sum()); overall_den += err.size
        if near_mask.any():
            near_num += float(err[near_mask].sum()); near_den += int(near_mask.sum()) * A
        if inner_mask.any():
            inner_num += float(err[inner_mask].sum()); inner_den += int(inner_mask.sum()) * A

    out = {
        "overall_mse": (overall_num / max(1, overall_den)),
        "near_mse": (near_num / max(1, near_den)) if near_den else float("nan"),
        "inner_mse": (inner_num / max(1, inner_den)) if inner_den else float("nan"),
        "per_episode": per_ep,
    }

    # print a compact report
    print(f"[{args.variant}] overall MSE: {out['overall_mse']:.6f} | "
          f"near-boundary: {out['near_mse']:.6f} | in-event: {out['inner_mse']:.6f}")

if __name__ == "__main__":
    main()
