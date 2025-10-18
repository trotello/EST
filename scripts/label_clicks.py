# scripts/label_clicks.py
from __future__ import annotations
import argparse, os, json, numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--episode-base", required=True, help="e.g., ep_taskA__traj001")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ev_path = os.path.join(args.work_dir, "events", args.model_key, args.episode_base + ".npz")
    if not os.path.isfile(ev_path):
        raise FileNotFoundError(ev_path)
    ev = np.load(ev_path)
    p = ev["b_prob"]; ts = ev["timestamps"]
    clicks = []

    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(ts, p, lw=1.5)
    ax.set_ylim(0,1); ax.set_xlabel("time (s)"); ax.set_ylabel("score")
    ax.set_title(args.episode_base)
    dot, = ax.plot([], [], "ro")

    def onclick(event):
        if event.inaxes != ax: return
        clicks.append(event.xdata)
        dot.set_data(clicks, [0.95]*len(clicks))
        fig.canvas.draw_idle()
        print(f"clicked at {event.xdata:.2f}s (total {len(clicks)})")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    out = args.out or os.path.join("labels", args.episode_base + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"fps": args.fps, "clicks_sec": clicks}, f, indent=2)
    print(f"saved {len(clicks)} clicks → {out}")

if __name__ == "__main__":
    main()
