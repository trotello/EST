# estlib/il/infer.py
# -*- coding: utf-8 -*-
"""
Inference utilities for BC policies at feature level.

Loads a saved policy (plain or EST), builds inputs x_t from cached features
(and EST sidecars if requested), and returns predicted actions per frame.
"""

from __future__ import annotations
import os, glob
import numpy as np
import torch
import json

from estlib.il.model import PolicyMLP
from estlib.data.libero_actions import read_actions_from_h5

def load_policy(policy_dir: str, device: str = "cpu") -> dict:
    """
    policy_dir: e.g., work/models/policy/bc_est_<model_key>/
    Returns a dict with 'model', 'use_est', 'input_dim', 'act_dim'
    """
    ckpt_path = os.path.join(policy_dir, "policy.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    data = None
    with open(os.path.join(policy_dir, "meta.json")) as json_file:
        data = json.load(json_file)
    print(list(ckpt.keys()))
    model = PolicyMLP(
        in_dim=data["input_dim"],
        act_dim=data["act_dim"],
        hidden=data["hidden"],
        layers=data["layers"],
    ).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    return {
        "model": model,
        "use_est": bool(data["use_est"]),
        "input_dim": int(data["input_dim"]),
        "act_dim": int(data["act_dim"]),
    }

def build_inputs_for_episode(features_npz: str, events_npz: str | None, use_est: bool) -> dict:
    """
    Returns:
      X: float32 [T, D (+3 if EST)]
      t: float32 [T]
      extras: dict with progress, boundary for plotting
    """
    f = np.load(features_npz)
    Z = f["z"].astype("float32")   # [T,D]
    t = f["t"].astype("float32") if "t" in f else np.arange(len(Z), dtype=np.float32)

    if not use_est:
        X = Z
        extras = {}
    else:
        if events_npz is None or not os.path.isfile(events_npz):
            # degenerate: EST requested but not available → fall back to zeros
            T = len(Z)
            progress = np.zeros((T,), dtype=np.float32)
            boundary = np.zeros((T,), dtype=np.float32)
        else:
            ev = np.load(events_npz)
            progress = ev["event_progress"].astype("float32")
            boundary = ev["boundary_hard"].astype("float32")
            T = min(len(Z), len(progress), len(boundary))
            Z = Z[:T]; t = t[:T]; progress = progress[:T]; boundary = boundary[:T]
        sinp = np.sin(2.0 * np.pi * progress)
        cosp = np.cos(2.0 * np.pi * progress)
        X = np.concatenate([Z, sinp[:,None], cosp[:,None], boundary[:,None]], axis=1)
        extras = {"progress": progress, "boundary_hard": boundary}
    return {"X": X, "t": t, "extras": extras}

@torch.no_grad()
def predict_episode(policy: PolicyMLP, X: np.ndarray, device: str = "cpu", batch: int = 4096) -> np.ndarray:
    """
    X: [T, input_dim] float32
    Returns a_hat: [T, A] float32
    """
    T = X.shape[0]
    out = []
    for i in range(0, T, batch):
        xb = torch.from_numpy(X[i:i+batch]).to(device).float()
        yb = policy(xb).cpu().numpy().astype("float32")
        out.append(yb)
    return np.concatenate(out, axis=0)
