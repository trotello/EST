# scripts/run_bocpd.py
# -*- coding: utf-8 -*-
"""
Compute BOCPD cp_prob[t] from per-episode prediction errors e_t.

Usage:
  python -m scripts.run_bocpd \
    --work-dir ./work \
    --model-key clip_clip-vit-base-patch16 \
    --hazard 0.02 --rmax 300
"""

from __future__ import annotations
import argparse
import glob
import os
from tqdm import tqdm
import numpy as np

from estlib.detect.bocpd import bocpd, BOCPDConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--hazard", type=float, default=0.02)   # ~1/expected_len_in_frames
    p.add_argument("--rmax", type=int, default=300)
    p.add_argument("--no-robust-z", action="store_true")
    args = p.parse_args()

    err_dir = os.path.join(args.work_dir, "errors", args.model_key)
    out_dir = os.path.join(args.work_dir, "bocpd", args.model_key)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(err_dir, "ep_*_err.npy")))
    if not files:
        raise FileNotFoundError(f"No error files found in {err_dir}. Run compute_pred_error first.")

    cfg = BOCPDConfig(
        hazard=args.hazard,
        rmax=args.rmax,
        use_robust_z=(not args.no_robust_z),
    )

    for f in tqdm(files, desc="bocpd"):
        e = np.load(f).astype("float32")
        cp = bocpd(e, cfg)  # [T] in [0,1]
        base = os.path.splitext(os.path.basename(f))[0].replace("_err", "")
        np.save(os.path.join(out_dir, f"{base}_bocpd.npy"), cp.astype("float32", copy=False))

if __name__ == "__main__":
    main()
