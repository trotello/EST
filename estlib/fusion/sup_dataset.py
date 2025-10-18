# estlib/fusion/sup_dataset.py
from __future__ import annotations
import os, glob, json
import numpy as np

def load_clicks(label_path: str) -> np.ndarray:
    with open(label_path, "r") as f:
        j = json.load(f)
    if "clicks_idx" in j: return np.array(j["clicks_idx"], dtype=int)
    assert "clicks_sec" in j and "fps" in j, "need clicks_sec + fps or clicks_idx"
    return (np.array(j["clicks_sec"], dtype=float) * float(j["fps"]) + 0.5).astype(int)

def make_frame_labels(T: int, clicks_idx: np.ndarray, pos_win: int, neg_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (y, mask) where:
      y[t] ∈ {0,1}, mask[t]∈{0,1} (1=keep this frame for training)
      positives: within ±pos_win frames of any click
      negatives: frames farther than neg_gap from any click (downsampled later)
    """
    y = np.zeros((T,), dtype=np.int64)
    pos = np.zeros((T,), dtype=bool)
    for c in clicks_idx:
        s = max(0, c - pos_win); e = min(T, c + pos_win + 1)
        y[s:e] = 1; pos[s:e] = True
    # candidate negatives
    neg = ~pos
    # mask selects all positives + a random subset of negatives (we’ll sample in trainer)
    mask = pos | neg
    return y, mask

def build_training_table(work_dir: str, model_key: str, labels_dir: str) -> list[dict]:
    """
    Returns a list of items:
      { "base": "ep_...", "T": T, "b": np.ndarray[T], "e": np.ndarray[T], "g": np.ndarray[T]|None, "y": np.ndarray[T], "mask": np.ndarray[T] }
    """
    feat_dir = os.path.join(work_dir, "features", model_key)
    err_dir  = os.path.join(work_dir, "errors",  model_key)
    boc_dir  = os.path.join(work_dir, "bocpd",   model_key)
    geb_dir  = os.path.join(work_dir, "gebd",    "scores")  # optional convention

    items = []
    for f in sorted(glob.glob(os.path.join(boc_dir, "ep_*_bocpd.npy"))):
        base = os.path.basename(f).replace("_bocpd.npy", "")
        label_path = os.path.join(labels_dir, base + ".json")
        if not os.path.isfile(label_path): continue

        b = np.load(f).astype("float32")
        e = np.load(os.path.join(err_dir, base + "_err.npy")).astype("float32")
        T = min(len(b), len(e))
        b = b[:T]; e = e[:T]
        g = None
        g_path = os.path.join(geb_dir, base + "_gebd.npy")
        if os.path.isfile(g_path):
            g = np.load(g_path).astype("float32")[:T]

        clicks = load_clicks(label_path)
        clicks = clicks[(clicks >= 0) & (clicks < T)]
        y, mask = make_frame_labels(T, clicks, pos_win=3, neg_gap=6)

        items.append({"base": base, "T": T, "b": b, "e": e, "g": g, "y": y, "mask": mask})
    return items
