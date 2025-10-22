# scripts/detect_events.py
# -*- coding: utf-8 -*-
"""
Fuse signals (BOCPD + pred-error [+ optional GEBD]) and export sidecars.

Usage:
  python -m scripts.detect_events \
    --work-dir ./work \
    --model-key clip_clip-vit-base-patch16 \
    --use-gebd 0 \
    --gebd-score-dir ./work/gebd/scores \
    --w-bocpd 0.5 --w-err 0.4 --w-gebd 0.1 \
    --smooth-window 9 --threshold 0.65 \
    --nms-win-sec 0.5 --min-dur-sec 1.0 \
    --fps 12
"""

from __future__ import annotations
import argparse
import glob
import os
import numpy as np
from tqdm import tqdm

from estlib.detect.gebd_runner import GebdRunner
from estlib.models.fusion_head import fuse_simple
from estlib.detect.postproc import smooth_prob, pick_boundaries, ids_and_progress
from estlib.models.fusion_head import FusionMLP
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--use-gebd", type=int, default=0)
    p.add_argument("--gebd-score-dir", default=None)

    p.add_argument("--w-bocpd", type=float, default=0.5)
    p.add_argument("--w-err", type=float, default=0.4)
    p.add_argument("--w-gebd", type=float, default=0.1)

    p.add_argument("--smooth-window", type=int, default=9)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--nms-win-sec", type=float, default=0.5)
    p.add_argument("--min-dur-sec", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--fusion", choices=["simple","mlp"], default="simple")
    p.add_argument("--fusion-model-dir", default=None)
    args = p.parse_args()

    feat_dir = os.path.join(args.work_dir, "features", args.model_key)
    err_dir  = os.path.join(args.work_dir, "errors",  args.model_key)
    boc_dir  = os.path.join(args.work_dir, "bocpd",   args.model_key)
    out_dir  = os.path.join(args.work_dir, "events",  args.model_key)
    os.makedirs(out_dir, exist_ok=True)

    fusion_model = None
    if args.fusion == "mlp":
        ck = torch.load(os.path.join(args.fusion_model_dir, "fusion_mlp.pt"), map_location="cpu")
        fusion_model = FusionMLP(in_dim=ck["in_dim"], hidden=ck["hidden"])
        fusion_model.load_state_dict(ck["state_dict"]); fusion_model.eval()

    # inside per-episode loop, right after you have b (bocpd), e (error), and optional g (gebd):

    # optional GEBD
    gebd = GebdRunner(
        mode="load" if args.use_gebd else "off",
        score_dir=args.gebd_score_dir
    )

    # iterate episodes by features (has timestamps T we may want)
    feat_files = sorted(glob.glob(os.path.join(feat_dir, "ep_*.npz")))
    if not feat_files:
        raise FileNotFoundError(f"No feature files at {feat_dir}")

    nms_win = int(round(args.nms_win_sec * args.fps))
    min_dur = int(round(args.min_dur_sec * args.fps))
    print("hey")

    for f in tqdm(feat_files, desc="detect_events"):
        base = os.path.splitext(os.path.basename(f))[0]          # ep_<...>
        print(base, f)
        ep_id = base.replace("ep_", "")

        # load timestamps length from features (small array)
        feat = np.load(f)
        T = int(feat["z"].shape[0])
        ts = feat["t"] if "t" in feat else np.arange(T) / args.fps

        # load signals; align by min length
        e_path  = os.path.join(err_dir,  f"{base}_err.npy")
        b_path  = os.path.join(boc_dir,  f"{base}_bocpd.npy")
        if not (os.path.isfile(e_path) and os.path.isfile(b_path)):
            # skip episodes without both signals
            continue
        e = np.load(e_path).astype("float32")
        b = np.load(b_path).astype("float32")
        L = min(T, len(e), len(b))
        e, b = e[:L], b[:L]
        ts = ts[:L]
        T = L

        # GEBD (optional)
        g = gebd.run(ep_id, T) if args.use_gebd else None
        if g is not None and len(g) != T:
            g = g[:T]

        if fusion_model is None:
            p = fuse_simple(b_bocpd=b, e_raw=e, b_gebd=g, w_bocpd=args.w_bocpd, w_err=args.w_err, w_gebd=args.w_gebd)
        else:
            feats = [b, e] + ([g] if (g is not None) else [])
            X = np.stack(feats, axis=1).astype("float32")  # [T,F]
            with torch.no_grad():
                p = fusion_model(torch.from_numpy(X)).numpy().astype("float32")

        # post-process
        p_s = smooth_prob(p, kind="median", window=args.smooth_window)
        cuts = pick_boundaries(p_s, thr=args.threshold, nms_win_frames=nms_win, min_duration_frames=min_dur)
        boundary_hard = np.zeros((T,), dtype=np.uint8)
        boundary_hard[cuts] = 1
        event_id, event_progress = ids_and_progress(T, cuts)

        # export sidecar
        out = os.path.join(out_dir, f"{base}.npz")
        np.savez_compressed(out,
            b_prob=p_s.astype("float32", copy=False),
            boundary_hard=boundary_hard,
            event_id=event_id,
            event_progress=event_progress,
            timestamps=ts.astype("float32", copy=False)
        )

if __name__ == "__main__":
    main()
