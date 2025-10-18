# estlib/features/clip_extractor.py
# -*- coding: utf-8 -*-
"""
CLIP-based per-frame embeddings using Hugging Face transformers.

Default model: openai/clip-vit-base-patch16
- Output: 512-D image embeddings (L2-normalized if normalize=True)
- Input:  RGB uint8 frames, any size; processor resizes to 224 with CLIP mean/std

Dependencies: torch, transformers, pillow, numpy
"""

from __future__ import annotations
from typing import List
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .base import FeatureExtractor

class ClipExtractor(FeatureExtractor):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        device: str = "cpu",
        batch_size: int = 64,
        normalize: bool = True,
    ):
        super().__init__(device=device, batch_size=batch_size, normalize=normalize)
        self.name = f"clip_{model_name.split('/')[-1]}"
        self.model_name = model_name

        self.processor = CLIPImageProcessor.from_pretrained(self.model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            self.model_name,
            use_safetensors=True,      # <- force .safetensors; no torch.load
            low_cpu_mem_usage=True,
        )
        self.model.eval().to(self.device)

        # projection_dim for ViT-B/16 is 512
        self.dim = int(self.model.config.projection_dim)

        # small safety for CPU-only Mac
        torch.set_grad_enabled(False)

    def _to_pil_list(self, frames: np.ndarray) -> List[Image.Image]:
        # frames: [B,H,W,3] uint8
        return [Image.fromarray(frames[i]) for i in range(frames.shape[0])]

    def encode_batch(self, frames: np.ndarray) -> np.ndarray:
        imgs = self._to_pil_list(frames)
        inputs = self.processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)  # [B,3,224,224]
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values)
            z = out.image_embeds  # [B, D], already projected
            if self.normalize:
                z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z.cpu().numpy().astype("float32", copy=False)
