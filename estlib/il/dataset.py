# estlib/il/dataset.py
# -*- coding: utf-8 -*-
"""
Feature-level BC dataset that aligns cached features (z[T,D]) with actions (a[T,A])
and injects EST tokens (sin/cos progress + boundary flag) when enabled.

Requires:
  - work/features/<model_key>/ep_*.npz   # z[T,D], t[T]
  - work/events/<model_key>/ep_*.npz     # event_progress[T], boundary_hard[T], timestamps[T]
  - LIBERO HDF5 under <dataset_root>/<split>/** for actions

If an episode has no matching HDF5, it is skipped.
"""

from __future__ import annotations
import glob, os, re
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

from estlib.data.libero_actions import read_actions_from_h5

def _rel_from_feature_fname(feat_npz: str) -> str:
    # ep_taskX__traj001.h5.npz -> taskX/traj001.h5   (we “flattened” subdirs with "__")
    base = os.path.basename(feat_npz)[3:-4]  # strip "ep_" + ".npz"
    return base.replace("__", os.sep)

class BCDataset(Dataset):
    def __init__(self,
                 work_dir: str,
                 model_key: str,
                 dataset_root: str,
                 split: str = "train",
                 use_est: bool = True,
                 max_episodes: Optional[int] = None):
        self.use_est = bool(use_est)
        self.features_dir = os.path.join(work_dir, "features", model_key)
        self.events_dir   = os.path.join(work_dir, "events",   model_key)
        self.dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
        self.split = split

        feat_files = sorted(glob.glob(os.path.join(self.features_dir, "ep_*.npz")))
        if max_episodes is not None:
            feat_files = feat_files[:max_episodes]

        self.entries: List[Tuple[str, str, str]] = []  # (feat_npz, events_npz, h5_path)
        split_dir = os.path.join(self.dataset_root, split)

        for f_npz in feat_files:
            base = os.path.basename(f_npz)
            e_npz = os.path.join(self.events_dir, base)
            if not os.path.isfile(e_npz):
                continue

            rel = _rel_from_feature_fname(f_npz)  # e.g., task/episode.ext
            stem = os.path.splitext(rel)[0]
            h5_path = None
            for ext in (".h5", ".hdf5"):
                cand = os.path.join(split_dir, stem + ext)
                if os.path.isfile(cand):
                    h5_path = cand
                    break
            if h5_path is None:
                continue
            self.entries.append((f_npz, e_npz, h5_path))

        if not self.entries:
            raise RuntimeError("No episodes found with features+events+HDF5 actions. Check paths.")

        # infer dims
        z0 = np.load(self.entries[0][0])["z"]; self.feat_dim = int(z0.shape[1])
        a0, _ = read_actions_from_h5(self.entries[0][2]); self.act_dim = int(a0.shape[1])
        self.extra_dim = 3 if self.use_est else 0
        self.input_dim = self.feat_dim + self.extra_dim

        # build per-episode alignment cache
        self._cache: List[Dict] = []
        for f_npz, e_npz, h5_path in self.entries:
            feat = np.load(f_npz)
            zT = int(feat["z"].shape[0])
            t_frames = feat["t"].astype("float32") if "t" in feat else np.arange(zT, dtype=np.float32)

            ev = np.load(e_npz)
            progress = ev["event_progress"].astype("float32")
            boundary = ev["boundary_hard"].astype("float32")

            L = min(zT, len(progress), len(boundary))
            t_frames = t_frames[:L]
            progress = progress[:L]
            boundary = boundary[:L]

            _, t_a = read_actions_from_h5(h5_path)
            idx = np.searchsorted(t_a, t_frames, side="left")
            idx = np.clip(idx, 0, len(t_a)-1)

            self._cache.append({
                "feat_path": f_npz,
                "events_path": e_npz,
                "h5_path": h5_path,
                "len": L,
                "a_index": idx.astype("int64"),
            })

        # flatten (episode, t) to a global index
        self._global = []
        for epi_id, meta in enumerate(self._cache):
            for t in range(meta["len"]):
                self._global.append((epi_id, t))

    def __len__(self) -> int: return len(self._global)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        epi_id, t = self._global[i]
        meta = self._cache[epi_id]

        z_all = np.load(meta["feat_path"])["z"]     # [T,D]
        z_t = z_all[t].astype("float32")

        ev = np.load(meta["events_path"])
        prog = float(ev["event_progress"][t])
        bh   = float(ev["boundary_hard"][t])

        acts, _ = read_actions_from_h5(meta["h5_path"])
        a_t = acts[ meta["a_index"][t] ].astype("float32")

        if self.use_est:
            sinp = np.sin(2.0 * np.pi * prog)
            cosp = np.cos(2.0 * np.pi * prog)
            x = np.concatenate([z_t, [sinp, cosp, bh]], axis=0)
        else:
            x = z_t

        return {"x": torch.from_numpy(x), "a": torch.from_numpy(a_t)}
