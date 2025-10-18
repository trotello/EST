# estlib/detect/pred_error.py
# -*- coding: utf-8 -*-
"""
Compute per-frame prediction error e_t for one episode:
  e_t = || fθ( (z_t - m)/s ) - (z_{t+1} - m)/s ||_2
Returns a length-T numpy array with e_0 copied from e_1 for alignment.
"""

from __future__ import annotations
import numpy as np
import torch

from estlib.models.forward_model import load_model, StatsNormalizer

@torch.no_grad()
def compute_pred_error_for_episode(z: np.ndarray, model_dir: str, device: str = "cpu") -> np.ndarray:
    """
    z: np.float32 [T, D]   (from your cached features)
    model_dir: path that contains forward_model.pt and meta.json
    """
    model, meta, norm = load_model(model_dir, device=device)
    model.eval(); norm.eval()

    Z = torch.from_numpy(z.astype("float32")).to(device)
    Zt    = Z[:-1]
    Ztp1  = Z[1:]
    zt_std   = norm(Zt)
    ztp1_std = norm(Ztp1)
    zhat_std = model(zt_std)
    e = torch.linalg.norm(zhat_std - ztp1_std, dim=-1)  # [T-1]
    # pad to length T
    e = torch.cat([e[:1], e], dim=0)
    return e.cpu().numpy().astype("float32", copy=False)

