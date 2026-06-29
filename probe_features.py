"""
probe_features.py  —  is the FROZEN front-end the bottleneck?
============================================================================
Question this answers: do the frozen ImageNet ResNet18 features already contain
enough to tell a dash from a non-dash (especially a jump)? If a trivial LINEAR
classifier on those features can separate them, the information is there and the
problem is downstream (head / threshold / hard negatives). If even with all the
training data a linear probe can't separate them, the information was thrown away
by the front-end -> fine-tuning the ResNet is justified.

Why a LINEAR probe: it's the weakest possible head (one matrix multiply, no
hidden layers, no time). It can only succeed if the classes are already roughly
linearly separable in feature space. So it's a clean test of "is the signal IN
the features", with the head's cleverness removed from the equation.

It reuses the SAME per-video .npz (features [n,512] + per-frame dash labels [n])
and the SAME by-video split as train.py, so there's no train/val leakage.

Run it wherever the .npz live (locally if you've extracted features here, or on
Kaggle pointing --features-dir at the cached dataset):

    python probe_features.py
    python probe_features.py --features-dir /kaggle/input/dash-features --topk 40
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dash_code import config as C
from dash_code import dataset as ds
from dash_code.labels import fmt_ts


# ---------------------------------------------------------------------------
# metrics (imbalance-robust; no sklearn dependency)
# ---------------------------------------------------------------------------
def roc_auc(scores, labels):
    """Probability a random dash frame outranks a random non-dash frame."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(scores, labels):
    """Area under the precision-recall curve (the right metric for rare events)."""
    order = np.argsort(-scores)
    y = (labels[order] > 0.5).astype(np.float64)
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    return float((precision * y).sum() / n_pos)


# ---------------------------------------------------------------------------
def load_frames(files):
    """Concatenate every frame of every video into X [N,512], y [N], plus a
    per-frame (video_name, frame_index) map so we can point at offenders later."""
    feats, labs, where = [], [], []
    for f in files:
        x, l = ds.load_npz(f)
        if x.shape[0] == 0:
            continue
        feats.append(x)
        labs.append(l)
        where.extend((f.stem, i) for i in range(x.shape[0]))
    if not feats:
        return None
    return np.concatenate(feats), np.concatenate(labs), where


def train_linear_probe(Xtr, ytr, epochs, device):
    """A single Linear(512 -> 1), pos-weighted BCE. The whole 'model' is one
    matrix + bias: if THIS separates the classes, the features already do."""
    in_dim = Xtr.shape[1]
    probe = nn.Linear(in_dim, 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)

    n_pos = float((ytr > 0.5).sum())
    n_neg = float((ytr <= 0.5).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X = torch.from_numpy(Xtr).to(device)
    y = torch.from_numpy(ytr).to(device).unsqueeze(1)
    bs = 8192
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = loss_fn(probe(X[idx]), y[idx])
            loss.backward()
            opt.step()
    return probe


def main():
    ap = argparse.ArgumentParser(description="Linear-probe the frozen features.")
    ap.add_argument("--features-dir", default=str(C.FEATURES_DIR),
                    help="folder of per-video .npz (default: processed/features)")
    ap.add_argument("--val-frac", type=float, default=C.VAL_FRAC)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--topk", type=int, default=30,
                    help="how many hard-negative frames to list for review")
    ap.add_argument("--no-dedup", action="store_true",
                    help="keep byte-identical duplicate clips (they cause train/val leakage)")
    args = ap.parse_args()

    fdir = Path(args.features_dir)
    files = sorted(fdir.glob("*.npz"))
    if not files:
        print(f"No .npz in {fdir}. Run Stage 2 (feature extraction) first, or "
              f"point --features-dir at the cached features (e.g. on Kaggle).")
        return

    if not args.no_dedup:
        import hashlib
        seen, unique = {}, []
        for p in files:
            feat, _ = ds.load_npz(p)
            h = hashlib.md5(np.ascontiguousarray(feat).tobytes()).hexdigest()
            if h in seen:
                print(f"  drop duplicate: {p.stem}  (== {seen[h]})")
            else:
                seen[h] = p.stem
                unique.append(p)
        print(f"deduplicated: {len(files)} -> {len(unique)} unique clips")
        files = unique

    train_files, val_files = ds.split_videos(files, args.val_frac, args.seed)
    print(f"videos: {len(files)} total -> {len(train_files)} train / {len(val_files)} val")

    tr = load_frames(train_files)
    va = load_frames(val_files)
    if tr is None or va is None:
        print("Not enough frames after split.")
        return
    Xtr, ytr, _ = tr
    Xva, yva, where_va = va

    # standardize using TRAIN stats only (z-score each of the 512 dims)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xva = ((Xva - mu) / sd).astype(np.float32)

    pos_tr = int((ytr > 0.5).sum())
    pos_va = int((yva > 0.5).sum())
    print(f"frames: {len(ytr)} train ({pos_tr} dash) / {len(yva)} val ({pos_va} dash)"
          f"  -> dash is {100*pos_va/len(yva):.2f}% of val frames")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = train_linear_probe(Xtr, ytr, args.epochs, device)

    with torch.no_grad():
        scores = probe(torch.from_numpy(Xva).to(device)).squeeze(1).cpu().numpy()

    auc = roc_auc(scores, yva)
    ap_ = average_precision(scores, yva)
    base = pos_va / len(yva)               # AP of random guessing = positive rate
    print("\n" + "=" * 60)
    print("LINEAR PROBE on frozen features  (val frames)")
    print("=" * 60)
    print(f"  ROC-AUC          : {auc:.3f}   (0.5 = chance, 1.0 = perfect rank)")
    print(f"  Average Precision: {ap_:.3f}   (random baseline = {base:.3f})")
    print(f"  AP lift over base: {ap_/base:.1f}x")

    # --- the diagnostic payload: the 'non-dash' frames the probe most wants to
    #     call a dash. Watch these. Jumps -> front-end can't separate (fine-tune).
    #     Missed real dashes -> label noise. Random/idle -> features just weak.
    neg = yva <= 0.5
    neg_idx = np.where(neg)[0]
    top = neg_idx[np.argsort(-scores[neg_idx])][:args.topk]
    print("\n" + "-" * 60)
    print(f"TOP {args.topk} HARD NEGATIVES (look dash-like but labelled non-dash)")
    print("Open these timestamps and ask: jump? missed dash? or nothing?")
    print("-" * 60)
    print(f"  {'video':<28} {'frame':>7} {'time':>10} {'score':>8}")
    rows = []
    for i in top:
        name, fr = where_va[i]
        t = fmt_ts(fr * C.MS_PER_FRAME)
        secs = round(fr * C.MS_PER_FRAME / 1000.0, 2)
        print(f"  {name[:28]:<28} {fr:>7} {t:>10} {scores[i]:>8.2f}")
        rows.append((name, fr, t, secs, round(float(scores[i]), 2)))

    # save the list so it doesn't scroll away — open this to find the moments
    import csv
    out_csv = C.PROJECT_ROOT / "predictions" / "hard_negatives.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "time_m_s_ms", "seconds", "score"])
        w.writerows(rows)
    print(f"\nsaved -> {out_csv}")
    print(f"(source clips are in {C.RAW_VIDEO_DIR}; 'seconds' column = where to seek)")

    print("\nReading the result:")
    print("  * High AUC/AP but these top negatives are JUMPS -> features mix")
    print("    jumps with dashes; the head can't fix what the front-end blurred")
    print("    -> fine-tune the ResNet (option A).")
    print("  * Low AUC/AP overall -> features barely encode dash at all")
    print("    -> front-end change is mandatory.")
    print("  * Top negatives are real dashes the labeler missed -> label noise.")


if __name__ == "__main__":
    main()
