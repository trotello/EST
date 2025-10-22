# estlib/models/forward_model.py
# -*- coding: utf-8 -*-
"""
Tiny forward model fθ: z_t -> ẑ_{t+1} + utilities.

- ForwardModel: 2–3 layer MLP (optionally residual)
- StatsNormalizer: per-dim mean/std standardization for latents
- save_model / load_model helpers write/read a .pt and a small meta.json

No YAML; ctor args map 1:1 to would-be config fields later.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- model ----------------------

class ForwardModel(nn.Module):
    """
    fθ: R^D -> R^D
    Simple MLP with optional residual connection to stabilize learning.
    """
    def __init__(self, d: int, hidden: int = 512, layers: int = 2, residual: bool = True):
        super().__init__()
        self.d = d
        self.residual = residual
        blocks = []
        in_dim = d
        for i in range(layers):
            blocks += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        blocks += [nn.Linear(hidden, d)]
        self.net = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        if self.residual and out.shape == z.shape:
            out = out + z
        return out

# ------------------- normalization -----------------

class StatsNormalizer(nn.Module):
    """
    Standardize features: (z - mean) / std  and invert at output side.
    Keeps mean, std as buffers; safe on cpu/cuda.
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6):
        super().__init__()
        m = torch.as_tensor(mean, dtype=torch.float32)
        s = torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("mean", m)
        self.register_buffer("std", torch.clamp(s, min=eps))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean) / self.std

    def inv(self, z_std: torch.Tensor) -> torch.Tensor:
        return z_std * self.std + self.mean

# ------------------- IO helpers --------------------

@dataclass
class ForwardMeta:
    dim: int
    hidden: int
    layers: int
    residual: bool
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    z_l2_normalized: bool = True  # from feature extractor
    loss: str = "l2"

def save_model(out_dir: str, model: ForwardModel, meta: ForwardMeta):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "forward_model.pt"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(asdict(meta), f, indent=2)

def load_model(model_dir: str, device: str = "cpu") -> tuple[ForwardModel, ForwardMeta, StatsNormalizer]:
    with open(os.path.join(model_dir, "meta.json"), "r") as f:
        meta_dict = json.load(f)
    meta = ForwardMeta(**meta_dict)
    model = ForwardModel(d=meta.dim, hidden=meta.hidden, layers=meta.layers, residual=meta.residual)
    model.load_state_dict(torch.load(os.path.join(model_dir, "forward_model.pt"), 
                                    map_location=device))
    model.to(device).eval()
    normalizer = StatsNormalizer(np.array(meta.mean, dtype=np.float32),
                                 np.array(meta.std, dtype=np.float32))
    normalizer.to(device).eval()
    return model, meta, normalizer

