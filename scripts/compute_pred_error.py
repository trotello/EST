# scripts/compute_pred_error.py
# -*- coding: utf-8 -*-
"""
Batch-compute prediction errors e_t for all episodes and save alongside features.

Usage:
  python -m scripts.compute_pred_error \
    --work-dir ./work \
    --model-key clip_clip-vit-base-patch16 \
    --device cpu
"""

from __future__ import annotations
import argparse
import glob
import os
from tqdm import tqdm
import numpy as np

from estlib.detect.pred_error import compute_pred_error_for_episode

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    feat_dir = os.path.join(args.work_dir, "features", args.model_key)
    model_dir = os.path.join(args.work_dir, "models", "forward", args.model_key)
    out_dir = os.path.join(args.work_dir, "errors", args.model_key)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(feat_dir, "ep_*.npz")))
    if not files:
        raise FileNotFoundError(f"No features found in {feat_dir}")

    for f in tqdm(files, desc="pred_error"):
        data = np.load(f)
        z = data["z"]
        e = compute_pred_error_for_episode(z, model_dir, device=args.device)  # [T]
        # save e next to features (same base name)
        base = os.path.splitext(os.path.basename(f))[0]
        np.save(os.path.join(out_dir, f"{base}_err.npy"), e)

if __name__ == "__main__":
    main()
