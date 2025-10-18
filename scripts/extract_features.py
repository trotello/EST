# scripts/extract_features.py
# -*- coding: utf-8 -*-
"""
Extract per-frame embeddings for all (or a subset of) episodes and cache them to disk.

Usage (no YAML):
  python -m scripts.extract_features \
      --root /PATH/TO/LIBERO/data \
      --split train \
      --work-dir ./work \
      --extractor clip \
      --model-name openai/clip-vit-base-patch16 \
      --device cpu \
      --target-fps 12 \
      --resize-short 224 \
      --crop 224 224 \
      --batch-size 64 \
      --max-episodes 50
"""

from __future__ import annotations
import os
import argparse
from tqdm import tqdm
import numpy as np

from estlib.data.libero import EpisodeIndex, EpisodeReader
from estlib.features.clip_extractor import ClipExtractor
# from estlib.features.dinov2_extractor import DinoV2Extractor   # enable later
from estlib.features.cache import save_cached, load_cached

def build_extractor(args):
    if args.extractor == "clip":
        return ClipExtractor(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            normalize=True,
        )
    elif args.extractor == "dino":
        from estlib.features.dinov2_extractor import DinoV2Extractor
        return DinoV2Extractor(
            arch=args.model_name, device=args.device, batch_size=args.batch_size, normalize=True
        )
    else:
        raise ValueError(f"Unknown extractor: {args.extractor}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="LIBERO dataset root")
    p.add_argument("--split", default="train")
    p.add_argument("--work-dir", default="./work")
    p.add_argument("--extractor", default="clip", choices=["clip","dino"])
    p.add_argument("--model-name", default="openai/clip-vit-base-patch16")
    p.add_argument("--device", default="cpu")  # "cuda" on Lambda
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-fps", type=float, default=12.0)
    p.add_argument("--resize-short", type=int, default=224)
    p.add_argument("--crop", type=int, nargs=2, metavar=("H","W"), default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    cache_dir = os.path.join(args.work_dir, "features")

    # 1) index & reader
    idx = EpisodeIndex(root=args.root, split=args.split)
    if args.max_episodes is not None:
        ep_list = [idx[i] for i in range(min(args.max_episodes, len(idx)))]
    else:
        ep_list = [idx[i] for i in range(len(idx))]

    reader = EpisodeReader(
        target_fps=args.target_fps,
        resize_short=args.resize_short,
        crop_to=tuple(args.crop) if args.crop else None,
    )

    # 2) extractor
    ext = build_extractor(args)
    model_key = ext.name  # used in cache path

    # 3) loop
    for epi in tqdm(ep_list, desc=f"extract:{model_key}"):
        # skip if cached (same model)
        cached = load_cached(cache_dir, model_key, epi.episode_id)
        if cached is not None:
            continue

        # read frames & ts
        sample = reader.load(epi)
        frames, ts = sample["frames"], sample["timestamps"]
        if len(frames) == 0:
            continue

        # encode
        z = ext.encode_episode(frames)   # [T, D]

        # save
        save_cached(cache_dir, model_key, epi.episode_id, z, ts)

if __name__ == "__main__":
    main()
