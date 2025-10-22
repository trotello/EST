# scripts/predict_actions.py
# -*- coding: utf-8 -*-
"""
Run a trained BC policy over all episodes that have features (+EST sidecars if needed)
and save predicted actions to work/preds/<variant>/<model_key>/ep_*.npy

Usage:
  python -m scripts.predict_actions \
    --work-dir ./work \
    --model-key clip_clip-vit-base-patch16 \
    --variant est \
    --dataset-root /PATH/TO/LIBERO/data \
    --split val \
    --device cpu \
    --max-episodes 50
"""

from __future__ import annotations
import argparse, os, glob
import numpy as np

from estlib.il.infer import load_policy, build_inputs_for_episode, predict_episode

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--variant", choices=["est","plain"], required=True)
    p.add_argument("--dataset-root", required=True)  # not used here directly but kept for symmetry
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    policy_dir = os.path.join(args.work_dir, "models", "policy", f"bc_{args.variant}_{args.model_key}")
    info = load_policy(policy_dir, device=args.device)
    use_est = info["use_est"]

    feat_dir = os.path.join(args.work_dir, "features", args.model_key)
    ev_dir   = os.path.join(args.work_dir, "events",   args.model_key)
    out_dir  = os.path.join(args.work_dir, "preds",    f"{args.variant}_{args.model_key}")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(feat_dir, "ep_*.npz")))
    if args.max_episodes: files = files[:args.max_episodes]

    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]     # ep_<...>
        epath = os.path.join(ev_dir, base + ".npz")
        pack = build_inputs_for_episode(f, epath, use_est=use_est)
        X, t = pack["X"], pack["t"]
        a_hat = predict_episode(info["model"], X, device=args.device)
        np.save(os.path.join(out_dir, base + "_ahat.npy"), a_hat.astype("float32"))
        np.save(os.path.join(out_dir, base + "_t.npy"), t.astype("float32"))

if __name__ == "__main__":
    main()
