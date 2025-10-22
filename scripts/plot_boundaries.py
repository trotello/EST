# scripts/plot_boundaries.py
# -*- coding: utf-8 -*-
"""
Quick diagnostic: plots fused score + hard boundaries for a few episodes.
"""
from __future__ import annotations
import os, glob, numpy as np, h5py, argparse, matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--max-plots", type=int, default=5)
    args = p.parse_args()

    ev_dir = os.path.join(args.work_dir, "events", args.model_key)
    files = sorted(glob.glob(os.path.join(ev_dir, "ep_*.npz")))[:args.max_plots]
    os.makedirs(os.path.join(ev_dir, "_plots"), exist_ok=True)

    for f in files:
        data = np.load(f)
        p = data["b_prob"]; bh = data["boundary_hard"]; ts = data["timestamps"]
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(ts, p, lw=1.5)
        where = np.where(bh > 0)[0]
        ax.vlines(ts[where], 0, 1, linestyles="dashed", alpha=0.5)
        ax.set_ylim(0,1); ax.set_xlabel("time (s)"); ax.set_ylabel("fused prob")
        ax.set_title(os.path.basename(f))
        out = os.path.join(ev_dir, "_plots", os.path.basename(f).replace(".npz", ".png"))
        fig.tight_layout(); fig.savefig(out); plt.close(fig)

if __name__ == "__main__":
    main()
