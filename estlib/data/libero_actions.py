# estlib/data/libero_actions.py
from __future__ import annotations
import re, numpy as np
try:
    import h5py
except Exception:
    h5py = None

def _find_candidate_action_paths(f: "h5py.File") -> list[str]:
    """Return a list of dataset paths that look like action arrays (T, A)."""
    cands = []
    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset): 
            return
        if obj.ndim != 2: 
            return
        T, A = obj.shape
        if T < 2 or A < 1 or A > 64:   # typical action dims are small-ish
            return
        low = name.lower()
        # strong matches first
        score = 0
        if re.search(r'(^|/)actions?($|/)', low): score += 3
        if low.endswith("/act") or low.endswith("/action"): score += 2
        if "action" in low: score += 1
        if "policy" in low or "traj" in low or "data" in low: score += 0.5
        if score > 0:
            cands.append((score, name))
    f.visititems(visit)
    cands.sort(key=lambda x: (-x[0], x[1]))
    return [name for _, name in cands]

def _find_timestamps_for_len(f: "h5py.File", T: int, default_fps: float) -> np.ndarray:
    """Try to locate a 1D timestamp array of length T; otherwise synthesize from fps attrs."""
    ts_keys = [
        "action_timestamps", "timestamps/actions", "time/actions",
        "timestamps", "time", "data/timestamps", "steps/timestamp"
    ]
    for k in ts_keys:
        if k in f:
            arr = f[k][...]
            if arr.ndim == 1 and len(arr) == T:
                return arr.astype("float32")
    fps = float(f.attrs.get("action_fps", f.attrs.get("fps", default_fps)))
    return np.arange(T, dtype=np.float32) / max(1e-6, fps)

def read_actions_from_h5(h5_path: str, default_fps: float = 20.0):
    """Return (actions[T,A] float32, timestamps[T] float32 in seconds)."""
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 action datasets.")
    with h5py.File(h5_path, "r") as f:
        # 1) strong exact keys (fast path)
        for k in ["actions", "data/actions", "observations/actions", "robot/actions"]:
            if k in f:
                acts = f[k][...].astype("float32")
                return acts, _find_timestamps_for_len(f, len(acts), default_fps)
        # 2) robust scan
        for path in _find_candidate_action_paths(f):
            acts = f[path][...].astype("float32")
            ts = _find_timestamps_for_len(f, len(acts), default_fps)
            return acts, ts
        # 3) give up
        raise KeyError(f"No action dataset found in {h5_path}")

