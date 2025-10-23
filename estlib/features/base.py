# estlib/features/base.py
# -*- coding: utf-8 -*-
"""
Abstract feature extractor API.

HYDRA NOTE: if I add Hydra later, ctor args (device, batch_size, normalize)
map 1:1 to config fields. For now, everything is passed as Python args.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class FeatureExtractor(ABC):
    """
    Base class for per-frame feature extractors.

    Contract:
      - encode_batch(frames): [B,H,W,3] uint8 RGB -> [B,D] float32
      - dim: embedding dimensionality (int)
      - name: model identifier (str)
    """
    name: str
    dim: int

    def __init__(self, device: str = "cpu", batch_size: int = 64, normalize: bool = True):
        self.device = device
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)

    @abstractmethod
    def encode_batch(self, frames: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def encode_episode(self, frames: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper to process a whole episode with batching.
        frames: [T,H,W,3] uint8 RGB
        returns z: [T,dim] float32
        """
        assert frames.ndim == 4 and frames.shape[-1] == 3 and frames.dtype == np.uint8
        T = frames.shape[0]
        out = []
        for i in range(0, T, self.batch_size):
            out.append(self.encode_batch(frames[i : i + self.batch_size]))
        return np.concatenate(out, axis=0).astype("float32", copy=False)
