# estlib/features/cache.py
# -*- coding: utf-8 -*-
"""
Disk cache helpers for episode embeddings.
Schema:
  work/features/<model_name>/ep_<safe_episode_id>.npz
    - z: float32 [T, D]
    - t: float32 [T] timestamps in seconds
"""

from __future__ import annotations
import os
import re
import numpy as np
from typing import Optional, Dict

def _safe_id(ep_id: str) -> str:
    # make episode_id safe for filenames (keep path-ish structure readable)
    s = re.sub(r"[^\w\-./]", "_", ep_id)
    return s.replace(os.sep, "__")  # flatten subdirs

def cache_path(cache_dir: str, model_name: str, episode_id: str) -> str:
    os.makedirs(os.path.join(cache_dir, model_name), exist_ok=True)
    fname = f"ep_{_safe_id(episode_id)}.npz"
    return os.path.join(cache_dir, model_name, fname)

def save_cached(cache_dir: str, model_name: str, episode_id: str, z: np.ndarray, t: np.ndarray) -> str:
    path = cache_path(cache_dir, model_name, episode_id)
    np.savez_compressed(path, z=z.astype("float32", copy=False), t=t.astype("float32", copy=False))
    return path

def load_cached(cache_dir: str, model_name: str, episode_id: str) -> Optional[Dict[str, np.ndarray]]:
    path = cache_path(cache_dir, model_name, episode_id)
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {"z": data["z"], "t": data["t"]}
