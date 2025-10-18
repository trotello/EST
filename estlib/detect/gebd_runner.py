# estlib/detect/gebd_runner.py
# -*- coding: utf-8 -*-
"""
GEBD runner wrapper.
- mode="off": returns zeros (useful on macOS CPU)
- mode="load": load precomputed per-frame GEBD scores from disk
  expected file: <score_dir>/ep_<episode_id>_gebd.npy   shape [T] in [0,1]
"""

from __future__ import annotations
import os
import numpy as np

class GebdRunner:
    def __init__(self, mode: str = "off", score_dir: str | None = None):
        self.mode = mode
        self.score_dir = score_dir

    def run(self, episode_id: str, T: int) -> np.ndarray:
        if self.mode == "off":
            return np.zeros((T,), dtype=np.float32)

        if self.mode == "load":
            if not self.score_dir:
                return np.zeros((T,), dtype=np.float32)
            path = os.path.join(self.score_dir, f"ep_{episode_id}_gebd.npy")
            if os.path.isfile(path):
                x = np.load(path).astype("float32", copy=False)
                if x.shape[0] == T:
                    return x
                # pad/trim to T
                y = np.zeros((T,), dtype=np.float32)
                m = min(T, x.shape[0])
                y[:m] = x[:m]
                return y
            return np.zeros((T,), dtype=np.float32)

        raise NotImplementedError(f"GEBD mode {self.mode!r} not implemented yet.")

