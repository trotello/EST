# scripts/train_fusion_mlp.py
from __future__ import annotations
import argparse, os, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from estlib.fusion.sup_dataset import build_training_table
from estlib.models.fusion_head import FusionMLP

class FrameSet(Dataset):
    def __init__(self, table: list[dict], use_gebd: bool = False, neg_ratio: float = 1.5):
        Xs, Ys = [], []
        for it in table:
            b, e, y, m = it["b"], it["e"], it["y"], it["mask"]
            feats = [b, e]
            if use_gebd and (it["g"] is not None): feats.append(it["g"])
            X = np.stack(feats, axis=1).astype("float32")  # [T, F]
            # sample: all positives + a subset of negatives
            pos_idx = np.where((y == 1) & m)[0]
            neg_idx = np.where((y == 0) & m)[0]
            if len(pos_idx) == 0: continue
            k_neg = int(min(len(neg_idx), neg_ratio * len(pos_idx)))
            if k_neg > 0:
                neg_sel = np.random.choice(neg_idx, size=k_neg, replace=False)
                idx = np.concatenate([pos_idx, neg_sel])
            else:
                idx = pos_idx
            Xs.append(X[idx]); Ys.append(y[idx].astype("float32"))
        self.X = np.concatenate(Xs, 0); self.Y = np.concatenate(Ys, 0)

    def __len__(self): return len(self.Y)
    def __getitem__(self, i): 
        return {"x": torch.from_numpy(self.X[i]), "y": torch.from_numpy(np.array(self.Y[i]))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--labels-dir", default="labels")
    ap.add_argument("--use-gebd", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    table = build_training_table(args.work-dir, args.model_key, args.labels_dir)  # noqa: E999 (hyphen)
    # argparse doesn't allow hyphens in var names; fix:
    table = build_training_table(args.work_dir, args.model_key, args.labels_dir)

    ds = FrameSet(table, use_gebd=bool(args.use_gebd))
    F = ds.X.shape[1]
    print(f"supervised fusion dataset: {len(ds)} frames, F={F}")

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=False)
    model = FusionMLP(in_dim=F, hidden=args.hidden).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCELoss()

    best = 1e9
    out_dir = os.path.join(args.work_dir, "models", "fusion", args.model_key)
    os.makedirs(out_dir, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train(); loss_sum = 0.0; n = 0
        for batch in dl:
            x = batch["x"].to(args.device).float()
            y = batch["y"].to(args.device).float()
            p = model(x)
            loss = bce(p, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * x.size(0); n += x.size(0)
        avg = loss_sum / max(1,n)
        print(f"[ep {ep}] BCE={avg:.4f}")
        if avg < best:
            best = avg
            torch.save({"state_dict": model.state_dict(), "in_dim": F, "hidden": args.hidden},
                       os.path.join(out_dir, "fusion_mlp.pt"))
            print(f"  ✓ saved best to {out_dir}")

if __name__ == "__main__":
    main()
