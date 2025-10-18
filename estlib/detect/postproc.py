# estlib/detect/postproc.py
# -*- coding: utf-8 -*-
"""
Turn a fused boundary score p[t] into hard cuts + event tokens.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter, maximum_filter1d

def smooth_prob(p: np.ndarray, kind: str = "median", window: int = 9) -> np.ndarray:
    if window <= 1: return p.astype("float32", copy=False)
    if kind == "median":
        # kernel size must be odd
        k = window if (window % 2 == 1) else window + 1
        return median_filter(p.astype("float32"), size=k, mode="nearest")
    raise ValueError(f"Unknown smooth kind: {kind}")

def pick_boundaries(p_smooth: np.ndarray,
                    thr: float = 0.65,
                    nms_win_frames: int = 6,
                    min_duration_frames: int = 12) -> np.ndarray:
    """
    p_smooth: [T] in [0,1] (already smoothed)
    Returns sorted indices of accepted boundaries.
    """
    T = len(p_smooth)
    if T == 0: return np.zeros((0,), dtype=np.int64)
    # threshold + local maxima (1D NMS)
    win = max(1, nms_win_frames)
    local_max = (p_smooth == maximum_filter1d(p_smooth, size=win, mode="nearest"))
    cand = np.where(local_max & (p_smooth >= thr))[0]
    # enforce minimum distance
    picks = []
    last = -10**9
    for t in cand:
        if t - last >= min_duration_frames:
            picks.append(int(t))
            last = t
    return np.array(picks, dtype=np.int64)

def ids_and_progress(T: int, cuts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    cuts: frame indices where a new event starts (hard boundaries)
    Returns:
      event_id[T] in {1..K}
      event_progress[T] in [0,1]
    """
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, T-1]
    eid  = np.zeros((T,), dtype=np.int32)
    prog = np.zeros((T,), dtype=np.float32)
    k = 1
    for s, e in zip(starts, ends):
        eid[s:e+1] = k
        L = max(1, e - s + 1)
        t = np.arange(s, e+1, dtype=np.float32)
        prog[s:e+1] = np.clip((t - s + 0.5) / L, 0.0, 1.0)
        k += 1
    return eid, prog
