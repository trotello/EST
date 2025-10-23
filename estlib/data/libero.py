# estlib/data/libero.py
# -*- coding: utf-8 -*-
"""
Episode discovery + reading for LIBERO-style demos.

This module is intentionally YAML-free. If I later add Hydra, the ctor args
map 1:1 to config fields (see "HYDRA NOTE" comments below).

It supports three common on-disk layouts:
  1) Video files (e.g., .mp4/.avi/.mov)
  2) Image directories with sequential .png/.jpg
  3) HDF5 files (robomimic/LIBERO style); we auto-discover an image dataset

Returns:
  - frames: uint8 [T, H, W, 3] in RGB
  - timestamps: float32 [T] in seconds (monotonic, inferred if needed)

Dependencies: numpy, opencv-python; optional: h5py for .h5/.hdf5
"""

from __future__ import annotations
import os
import re
import glob
import json
import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2

try:
    import h5py  # optional; only needed for HDF5
except Exception:
    h5py = None




_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_VID_EXTS = (".mp4", ".avi", ".mov", ".mkv")
_H5_EXTS  = (".h5", ".hdf5")


def _natural_key(path: str):
    """Sort key that treats numbers numerically (e.g., img2 < img10)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(path))]


def _ensure_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """cv2 reads BGR; convert to RGB."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR frame, got {frame_bgr.shape}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _center_crop(frames: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Center-crop [T,H,W,3] to (out_h, out_w)."""
    T, H, W, C = frames.shape
    if H == out_h and W == out_w:
        return frames
    top  = max(0, (H - out_h) // 2)
    left = max(0, (W - out_w) // 2)
    return frames[:, top:top+out_h, left:left+out_w, :]


def _resize_short_side(frames: np.ndarray, short: int) -> np.ndarray:
    """Resize so the shorter spatial side becomes `short`, preserve aspect."""
    T, H, W, C = frames.shape
    if min(H, W) == short:
        return frames
    if H < W:
        new_h, new_w = short, int(round(W * (short / H)))
    else:
        new_w, new_h = short, int(round(H * (short / W)))
    out = np.empty((T, new_h, new_w, C), dtype=frames.dtype)
    for i in range(T):
        out[i] = cv2.resize(frames[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out


def _resample_by_fps(num_src: int, fps_src: float, fps_tgt: float) -> np.ndarray:
    """
    Index mapping from a source frame grid (0..num_src-1 @ fps_src) to a target
    grid at fps_tgt using nearest-neighbor in time.
    """
    if fps_src <= 0 or fps_tgt <= 0:
        # fall back to identity if fps unknown
        return np.arange(num_src, dtype=np.int64)
    if num_src == 0:
        return np.empty((0,), dtype=np.int64)
    dur = (num_src - 1) / fps_src
    n_tgt = int(math.floor(dur * fps_tgt)) + 1
    t_src = np.arange(num_src, dtype=np.float32) / fps_src
    t_tgt = np.arange(n_tgt, dtype=np.float32) / fps_tgt
    idx   = np.searchsorted(t_src, t_tgt, side="left")
    idx   = np.clip(idx, 0, num_src - 1)
    # snap to nearest (optional refinement)
    left_ok  = idx > 0
    right_ok = idx < num_src - 1
    left_d   = np.full_like(idx, np.inf, dtype=np.float32)
    right_d  = np.full_like(idx, np.inf, dtype=np.float32)
    left_d[left_ok]   = np.abs(t_tgt[left_ok] - t_src[idx[left_ok] - 1])
    right_d[right_ok] = np.abs(t_tgt[right_ok] - t_src[idx[right_ok]])
    use_left = left_ok & (left_d < right_d)
    idx[use_left] -= 1
    return idx.astype(np.int64)


def _maybe_h5_image_key(h5: "h5py.File") -> Optional[str]:
    """
    Try to discover an image dataset key in a LIBERO/robomimic-style file.
    We probe a few common paths and then fall back to the first HxWx3 or 3xHxW.
    """
    #change later when I know what my files are
    candidates = [
        "observations/images/agentview_rgb",
        "observations/images/agentview",
        "observations/images/rgb",
        "data/observations/images/agentview",
        "data/observations/images/rgb",
        "images",
    ]
    for k in candidates:
        if k in h5:
            return k
    # heuristic: find any dataset with last dim 3 or first dim 3 and len(shape)>=3
    best = None
    def _walk(name, obj):
        nonlocal best
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
            shape = obj.shape
            if shape[-1] == 3 or shape[1] == 3:
                best = name
    h5.visititems(_walk)
    return best


@dataclass
class EpisodeInfo:
    episode_id: str
    path: str
    storage: str      # "video" | "images" | "hdf5"
    split: str
    length_frames: Optional[int] = None
    extra: Dict = None


class EpisodeIndex:
    """
    Scans a LIBERO root directory and collects episode descriptors.
    We look under <root>/<split>/** for:
      - video files (*.mp4/*.avi/*.mov/*.mkv)
      - image folders (containing > 0 images)
      - HDF5 files (*.h5/*.hdf5)
    """
    def __init__(self, root: str, split: str = "train"):
        # HYDRA NOTE: these args would be data.root and data.split later.
        self.root = os.path.abspath(os.path.expanduser(root))
        self.split = split
        self._episodes: List[EpisodeInfo] = []
        self._scan()

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int) -> EpisodeInfo:
        return self._episodes[idx]

    def info(self, idx: int) -> Dict:
        e = self._episodes[idx]
        return {
            "episode_id": e.episode_id,
            "path": e.path,
            "storage": e.storage,
            "split": e.split,
            "length_frames": e.length_frames,
            "extra": e.extra or {},
        }

    def _scan(self):
        base = os.path.join(self.root, self.split)
        if not os.path.isdir(base):
            warnings.warn(f"[EpisodeIndex] Split folder not found: {base}")
            return

        # 1) HDF5 files
        for ext in _H5_EXTS:
            for p in glob.glob(os.path.join(base, "**", f"*{ext}"), recursive=True):
                eid = os.path.relpath(p, base)
                self._episodes.append(EpisodeInfo(
                    episode_id=eid, path=p, storage="hdf5", split=self.split))

        # 2) Videos
        for ext in _VID_EXTS:
            for p in glob.glob(os.path.join(base, "**", f"*{ext}"), recursive=True):
                eid = os.path.relpath(p, base)
                self._episodes.append(EpisodeInfo(
                    episode_id=eid, path=p, storage="video", split=self.split))

        # 3) Image directories (contain at least one image file)
        for d in glob.glob(os.path.join(base, "**"), recursive=True):
            if not os.path.isdir(d):
                continue
            imgs = [f for f in os.listdir(d) if f.lower().endswith(_IMG_EXTS)]
            if len(imgs) > 0:
                eid = os.path.relpath(d, base)
                self._episodes.append(EpisodeInfo(
                    episode_id=eid, path=d, storage="images", split=self.split))

        # Deduplicate (in case a directory contains both video and images)
        unique = {}
        for e in self._episodes:
            unique_key = (e.episode_id, e.storage)
            if unique_key not in unique:
                unique[unique_key] = e
        self._episodes = list(unique.values())


class EpisodeReader:
    """
    Reads an episode into (frames RGB uint8 [T,H,W,3], timestamps float32 [T]).
    Handles video files, image directories, and HDF5 files.

    Args:
        target_fps:   desired output fps (use <= 16 to keep CPU friendly)
        resize_short: resize the shorter side to this many pixels (0 = skip)
        crop_to:      optional explicit (H, W) center-crop after resizing
        default_src_fps_for_images: assumed fps when reading image folders
    """
    def __init__(
        self,
        target_fps: float = 12.0,                # HYDRA NOTE: data.fps
        resize_short: int = 224,                 # HYDRA NOTE: data.resize_short
        crop_to: Optional[Tuple[int, int]] = None,
        default_src_fps_for_images: float = 30.0
    ):
        print(target_fps)
        self.target_fps = float(target_fps)
        self.resize_short = int(resize_short) if resize_short else 0
        self.crop_to = crop_to
        self.default_img_fps = float(default_src_fps_for_images)


    def load(self, info: EpisodeInfo) -> Dict[str, np.ndarray]:
        if info.storage == "video":
            frames, ts, fps_src = self._read_video(info.path)
        elif info.storage == "images":
            frames, ts, fps_src = self._read_image_dir(info.path)
        elif info.storage == "hdf5":
            frames, ts, fps_src = self._read_hdf5(info.path)
        else:
            raise ValueError(f"Unknown storage type: {info.storage}")

        # resample to target fps
        idx = _resample_by_fps(len(frames), fps_src, self.target_fps)
        frames = frames[idx]
        if ts is not None and len(ts) == len(frames):
            timestamps = ts[idx]
        else:
            # infer timestamps from target fps
            timestamps = np.arange(len(frames), dtype=np.float32) / self.target_fps

        # resize & optional crop
        if self.resize_short and min(frames.shape[1], frames.shape[2]) != self.resize_short:
            frames = _resize_short_side(frames, self.resize_short)
        if self.crop_to is not None:
            frames = _center_crop(frames, self.crop_to[0], self.crop_to[1])

        return {
            "frames": frames.astype(np.uint8, copy=False),
            "timestamps": timestamps.astype(np.float32, copy=False),
            "meta": {
                "episode_id": info.episode_id,
                "source_path": info.path,
                "storage": info.storage,
                "src_fps": fps_src,
                "tgt_fps": self.target_fps,
                "resize_short": self.resize_short,
                "crop_to": self.crop_to,
            }
        }

    def _read_video(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {path}")
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for _ in range(n_frames):
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(_ensure_rgb(bgr))
        cap.release()
        frames = np.stack(frames, axis=0) if frames else np.zeros((0, 1, 1, 3), dtype=np.uint8)
        # Some containers lie about FPS. If invalid, fall back to default image fps.
        if not fps_src or not np.isfinite(fps_src) or fps_src <= 0:
            fps_src = self.default_img_fps
        # Timestamps direct from FPS (videos rarely store per-frame PTS via cv2)
        timestamps = np.arange(len(frames), dtype=np.float32) / float(fps_src)
        return frames, timestamps, float(fps_src)

    def _read_image_dir(self, dir_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                 if f.lower().endswith(_IMG_EXTS)]
        files.sort(key=_natural_key)
        frames = []
        for f in files:
            bgr = cv2.imread(f, cv2.IMREAD_COLOR)
            if bgr is None:
                warnings.warn(f"[EpisodeReader] Failed to read image: {f}")
                continue
            frames.append(_ensure_rgb(bgr))
        frames = np.stack(frames, axis=0) if frames else np.zeros((0, 1, 1, 3), dtype=np.uint8)
        # We don’t know the real fps of image sequences; assume default.
        fps_src = self.default_img_fps
        timestamps = np.arange(len(frames), dtype=np.float32) / fps_src
        return frames, timestamps, fps_src

    def _read_hdf5(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        if h5py is None:
            raise ImportError("h5py is not installed, but required to read HDF5 episodes.")
        with h5py.File(path, "r") as h5:
            key = _maybe_h5_image_key(h5)
            if key is None:
                raise KeyError(f"No image dataset found in {path}")
            data = h5[key][...]  # load into memory
            # Normalize shapes to [T,H,W,3] RGB
            if data.ndim == 4:
                if data.shape[-1] == 3:
                    frames = data
                elif data.shape[1] == 3:  # [T,3,H,W] -> [T,H,W,3]
                    frames = np.transpose(data, (0, 2, 3, 1))
                else:
                    raise ValueError(f"Unexpected HDF5 image shape {data.shape} at {key}")
            else:
                raise ValueError(f"Expected 4D images, got shape {data.shape} at {key}")
            # dtype normalization
            if frames.dtype != np.uint8:
                # assume float in [0,1] or [0,255], clip and cast
                fr = frames.astype(np.float32)
                if fr.max() <= 1.0:
                    fr = np.clip(fr * 255.0, 0, 255)
                frames = fr.astype(np.uint8)

            # timestamps (optional). Try to read if present; else infer later.
            ts = None
            for cand in ("timestamps", "time", "meta/timestamps"):
                if cand in h5:
                    ts = h5[cand][...].astype(np.float32)
                    break

            # fps: try attrs; else fall back to default image fps
            fps_src = float(h5.attrs.get("fps", self.default_img_fps))
        return frames, ts, fps_src


'''
minimal usage:
from estlib.data.libero import EpisodeIndex, EpisodeReader

idx = EpisodeIndex(root="/PATH/TO/LIBERO/data", split="train")
reader = EpisodeReader(target_fps=12, resize_short=224, crop_to=(224,224))

print("Found episodes:", len(idx))
sample = reader.load(idx[0])
print(sample["frames"].shape, sample["timestamps"][:5], sample["meta"])
'''