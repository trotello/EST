# scripts/plot_actions.py
# -*- coding: utf-8 -*-
"""
Plot GT vs predicted actions for a few episodes with boundary markers.

Usage:
  python -m scripts.plot_actions \
    --work-dir ./work \
    --model-key clip_clip-vit-base-patch16 \
    --variant est \
    --max-plots 6
"""

from __future__ import annotations
import argparse, os, glob
import numpy as np
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--variant", choices=["est","plain"], required=True)
    p.add_argument("--max-plots", type=int, default=6)
    args = p.parse_args()

    preds_dir = os.path.join(args.work_dir, "preds", f"{args.variant}_{args.model_key}")
    ev_dir    = os.path.join(args.work_dir, "events", args.model_key)
    files = sorted(glob.glob(os.path.join(preds_dir, "ep_*_ahat.npy")))[:args.max_plots]
    out_dir = os.path.join(preds_dir, "_plots"); os.makedirs(out_dir, exist_ok=True)

    for pf in files:
        base = os.path.basename(pf).replace("_ahat.npy", "")
        a_hat = np.load(pf); t = np.load(os.path.join(preds_dir, base + "_t.npy"))
        evp = os.path.join(ev_dir, base + ".npz")
        bh = np.zeros((len(a_hat),), dtype=int)
        if os.path.isfile(evp):
            bh = np.load(evp)["boundary_hard"][:len(a_hat)]
        A = a_hat.shape[1]
        fig, ax = plt.subplots(A, 1, figsize=(10, 2.0*A), sharex=True)
        if A == 1: ax = [ax]
        for a in range(A):
            ax[a].plot(t, a_hat[:,a], label="pred", linewidth=1.2)
            ax[a].set_ylabel(f"a{a}")
            # vertical lines at boundaries
            where = np.where(bh > 0)[0]
            for w in where:
                ax[a].axvline(t[w], color="k", linestyle="--", alpha=0.3)
        ax[-1].set_xlabel("time (s)")
        fig.suptitle(base)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, base + ".png")); plt.close(fig)

if __name__ == "__main__":
    main()
