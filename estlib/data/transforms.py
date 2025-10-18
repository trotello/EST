# estlib/data/transforms.py
# -*- coding: utf-8 -*-
"""
Lightweight video transforms used by EpisodeReader and (later) by feature extraction.
Kept separate so you can reuse them in training pipelines.
"""

from __future__ import annotations
import math
import numpy as np
import cv2

def resize_short_side(frames: np.ndarray, short: int) -> np.ndarray:
    """Resize [T,H,W,3] so min(H,W) == short, preserving aspect."""
    T, H, W, C = frames.shape
    if min(H, W) == short:
        return frames
    if H < W:
        new_h, new_w = short, int(round(W * (short / H)))
    else:
        new_w, new_h = short, int(round(H * (short / W)))
    out = np.empty((T, new_h, new_w, C), dtype=frames.dtype)
    for i in range(T):
        out[i] = cv2.resize(frames[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out

def center_crop(frames: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Center-crop [T,H,W,3] to (out_h, out_w)."""
    T, H, W, C = frames.shape
    if H == out_h and W == out_w:
        return frames
    top  = max(0, (H - out_h) // 2)
    left = max(0, (W - out_w) // 2)
    return frames[:, top:top+out_h, left:left+out_w, :]

def resample_by_fps(num_src: int, fps_src: float, fps_tgt: float) -> np.ndarray:
    """
    Compute index mapping from src frames @ fps_src to target fps_tgt.
    Returns int64 indices into the source frame array.
    """
    if fps_src <= 0 or fps_tgt <= 0:
        return np.arange(num_src, dtype=np.int64)
    if num_src == 0:
        return np.empty((0,), dtype=np.int64)
    dur = (num_src - 1) / fps_src
    n_tgt = int(math.floor(dur * fps_tgt)) + 1
    t_src = np.arange(num_src, dtype=np.float32) / fps_src
    t_tgt = np.arange(n_tgt, dtype=np.float32) / fps_tgt
    idx   = np.searchsorted(t_src, t_tgt, side="left")
    idx   = np.clip(idx, 0, num_src - 1)
    left_ok  = idx > 0
    right_ok = idx < num_src - 1
    left_d   = np.full_like(idx, np.inf, dtype=np.float32)
    right_d  = np.full_like(idx, np.inf, dtype=np.float32)
    left_d[left_ok]   = np.abs(t_tgt[left_ok] - t_src[idx[left_ok] - 1])
    right_d[right_ok] = np.abs(t_tgt[right_ok] - t_src[idx[right_ok]])
    use_left = left_ok & (left_d < right_d)
    idx[use_left] -= 1
    return idx.astype(np.int64)
