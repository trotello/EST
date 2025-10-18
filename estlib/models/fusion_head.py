# estlib/models/fusion_head.py
# -*- coding: utf-8 -*-
"""
Signal fusion:
- start with a label-free weighted sum after robust normalization
- keep an optional tiny MLP stub for when you add timestamp clicks later
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

def _robust_norm(x: np.ndarray, lo_p: float = 5.0, hi_p: float = 95.0) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_p, hi_p])
    rng = max(1e-6, hi - lo)
    y = (x - lo) / rng
    return np.clip(y, 0.0, 1.0).astype("float32", copy=False)

def fuse_simple(b_bocpd: np.ndarray,
                e_raw: np.ndarray,
                b_gebd: np.ndarray | None = None,
                w_bocpd: float = 0.5,
                w_err: float = 0.4,
                w_gebd: float = 0.1) -> np.ndarray:
    """
    Robust-percentile normalize each track to [0,1], then weighted sum.
    """
    g = _robust_norm(b_bocpd)
    r = _robust_norm(e_raw)
    if b_gebd is None:
        s = 0.0
        w_gebd = 0.0
    else:
        s = _robust_norm(b_gebd)
    wsum = max(1e-6, w_bocpd + w_err + w_gebd)
    p = (w_bocpd * g + w_err * r + w_gebd * s) / wsum
    return p.astype("float32", copy=False)

# ---- (optional) tiny MLP if you add a handful of timestamp labels later ----

class FusionMLP(nn.Module):
    """
    2-layer MLP over per-frame features [b_bocpd, e_raw, b_gebd, d_brightness, flow_mag]
    """
    def __init__(self, in_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [T, F]
        return self.net(x).squeeze(-1)                    # [T]
