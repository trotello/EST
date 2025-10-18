# estlib/features/dinov2_extractor.py
# -*- coding: utf-8 -*-
"""
DINOv2 extractor (stub). Enable when you want to compare to CLIP.
Requires: torch>=2, torchvision, and either timm or torch.hub for dinov2 weights.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import FeatureExtractor

class DinoV2Extractor(FeatureExtractor):
    def __init__(self, arch: str = "dinov2_vits14", device: str = "cpu", batch_size: int = 64, normalize: bool = True):
        super().__init__(device=device, batch_size=batch_size, normalize=normalize)
        self.name = arch
        # pull weights via torch.hub (first run downloads)
        self.model = torch.hub.load("facebookresearch/dinov2", arch).to(self.device).eval()
        # vit-s/14 = 384-D global avg-pooled tokens by default (check model specifics)
        self.dim = 384
        torch.set_grad_enabled(False)

    def _pre(self, frames: np.ndarray) -> torch.Tensor:
        # minimal preprocessing to 224, [0,1] -> normalized by ImageNet mean/std
        from torchvision import transforms as T
        tfm = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])
        imgs = [tfm(Image.fromarray(fr)) for fr in frames]
        return torch.stack(imgs, dim=0)  # [B,3,224,224]

    def encode_batch(self, frames: np.ndarray) -> np.ndarray:
        x = self._pre(frames).to(self.device)
        with torch.no_grad():
            feats = self.model(x)           # model-dependent; many dinov2 weights return token features
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            # global pooling if tokens: [B, tokens, C] -> [B, C]
            if feats.ndim == 3:
                feats = feats.mean(dim=1)
            z = feats
            if self.normalize:
                z = F.normalize(z, p=2, dim=-1)
        return z.cpu().numpy().astype("float32", copy=False)
