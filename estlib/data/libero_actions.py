# estlib/data/libero_actions.py
from __future__ import annotations
import os
import numpy as np
try:
    import h5py
except Exception:
    h5py = None

_ACTION_KEYS = ["actions", "data/actions", "observations/actions", "robot/actions"]
_TS_KEYS = ["action_timestamps", "timestamps/actions", "time/actions", "timestamps", "time"]

def read_actions_from_h5(h5_path: str, default_fps: float = 20.0):
    """Return (actions[T,A] float32, timestamps[T] float32 in seconds)."""
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 action logs.")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        acts = None
        for k in _ACTION_KEYS:
            if k in f:
                acts = f[k][...]; break
        if acts is None:
            # heuristic fallback: first 2D float dataset
            for name, ds in f.items():
                if hasattr(ds, "shape") and len(ds.shape) == 2 and np.issubdtype(ds.dtype, np.floating):
                    acts = ds[...]; break
        if acts is None:
            raise KeyError(f"No action dataset found in {h5_path}")
        actions = acts.astype("float32")

        ts = None
        for k in _TS_KEYS:
            if k in f:
                arr = f[k][...]
                if arr.ndim == 1 and len(arr) == len(actions):
                    ts = arr.astype("float32"); break
        if ts is None:
            fps = float(f.attrs.get("action_fps", f.attrs.get("fps", default_fps)))
            ts = np.arange(len(actions), dtype=np.float32) / max(1e-6, fps)

    return actions, ts
