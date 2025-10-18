# scripts/train_forward_model.py
# -*- coding: utf-8 -*-
"""
Train the tiny forward model fθ: z_t -> ẑ_{t+1}.

Usage:
  python -m scripts.train_forward_model \
    --work-dir ./work \
    --model-key clip_clip-vit-base-patch16 \
    --hidden 512 --layers 2 --residual 1 \
    --batch-size 1024 --epochs 3 --lr 3e-4 \
    --max-pairs-per-file 5000 \
    --device cpu
"""

from __future__ import annotations
import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from estlib.models.forward_model import ForwardModel, ForwardMeta, save_model, StatsNormalizer
from estlib.models.data_pairs import FeaturePairDataset, compute_mean_std

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--model-key", required=True, help="folder name under work/features/")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--residual", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-pairs-per-file", type=int, default=5000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    features_dir = os.path.join(args.work_dir, "features")
    out_dir = os.path.join(args.work_dir, "models", "forward", args.model_key)
    os.makedirs(out_dir, exist_ok=True)

    # compute mean/std over all features (even if CLIP is L2-normalized, this helps)
    mean, std = compute_mean_std(features_dir, args.model_key)
    # build dataset/loader with standardization
    ds = FeaturePairDataset(features_dir, args.model_key, max_pairs_per_file=args.max_pairs_per_file,
                            mean=mean, std=std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    # infer embedding dim from one sample
    sample = next(iter(dl))
    D = sample["z_t"].shape[-1]
    model = ForwardModel(d=D, hidden=args.hidden, layers=args.layers, residual=bool(args.residual))
    model.to(args.device)
    norm = StatsNormalizer(mean, std).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in tqdm(dl, desc=f"train fwd ep{ep}/{args.epochs}"):
            zt = batch["z_t"].to(args.device)
            ztp1 = batch["z_tp1"].to(args.device)
            # (z-mean)/std
            zt_std   = norm(zt)
            ztp1_std = norm(ztp1)
            zhat_std = model(zt_std)
            loss = torch.mean((zhat_std - ztp1_std) ** 2)  # L2 on standardized space
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * zt.shape[0]
            n += zt.shape[0]
        epoch_loss = running / max(1, n)
        print(f"[epoch {ep}] train MSE: {epoch_loss:.6f}")
        if epoch_loss < best:
            best = epoch_loss
            meta = ForwardMeta(dim=D, hidden=args.hidden, layers=args.layers, residual=bool(args.residual),
                               mean=tuple(float(x) for x in mean), std=tuple(float(x) for x in std))
            save_model(out_dir, model, meta)
            print(f"  saved best to {out_dir}")

if __name__ == "__main__":
    main()
