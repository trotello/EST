# estlib/models/data_pairs.py
# -*- coding: utf-8 -*-
"""
Dataset and loader that stream (z_t -> z_{t+1}) pairs from cached feature NPZs.

- Scans work/features/<model_key>/*.npz
- Optionally subsamples long episodes to limit epoch size
- Builds pairs on the fly to avoid huge RAM use
"""

from __future__ import annotations
import glob
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

class PairIndex:
    """
    Light index mapping a global pair id -> (file_path, local_index_t).
    """
    def __init__(self, npz_files: List[str], max_pairs_per_file: Optional[int] = None):
        self.entries: List[Tuple[str, int]] = []
        for f in npz_files:
            data = np.load(f)
            T = int(data["z"].shape[0])
            n_pairs = max(0, T - 1)
            if n_pairs == 0: 
                continue
            if max_pairs_per_file is not None and n_pairs > max_pairs_per_file:
                # uniform subsample indices
                idx = np.linspace(0, n_pairs - 1, num=max_pairs_per_file, dtype=int)
            else:
                idx = np.arange(n_pairs, dtype=int)
            for t in idx.tolist():
                self.entries.append((f, t))

    def __len__(self): return len(self.entries)
    def get(self, i: int) -> Tuple[str, int]: return self.entries[i]

class FeaturePairDataset(Dataset):
    def __init__(self, features_dir: str, model_key: str, max_pairs_per_file: Optional[int] = None,
                 mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        pattern = os.path.join(features_dir, model_key, "ep_*.npz")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No feature files found at {pattern}")
        self.index = PairIndex(files, max_pairs_per_file=max_pairs_per_file)
        self.mean = mean
        self.std = std

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        fpath, t = self.index.get(i)
        data = np.load(fpath)
        z = data["z"]  # [T,D]
        zt = z[t].astype("float32")
        ztp1 = z[t + 1].astype("float32")
        if self.mean is not None and self.std is not None:
            zt = (zt - self.mean) / self.std
            ztp1 = (ztp1 - self.mean) / self.std
        return {
            "z_t": torch.from_numpy(zt),       # [D]
            "z_tp1": torch.from_numpy(ztp1),   # [D]
        }

def compute_mean_std(features_dir: str, model_key: str, clip: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-dim mean/std over all z’s (robust: clip to ±clip*std iteratively).
    """
    pattern = os.path.join(features_dir, model_key, "ep_*.npz")
    files = sorted(glob.glob(pattern))
    zs = []
    for f in files:
        z = np.load(f)["z"].astype("float32")
        zs.append(z)
    Z = np.concatenate(zs, axis=0)  # [N,D]
    # robust mean/std
    m = Z.mean(axis=0)
    s = Z.std(axis=0) + 1e-6
    # optional winsorization
    Z = np.clip(Z, m - clip * s, m + clip * s)
    m = Z.mean(axis=0)
    s = Z.std(axis=0) + 1e-6
    return m.astype("float32"), s.astype("float32")
