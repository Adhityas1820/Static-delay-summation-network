"""
train.py  —  STAGE 3 of the DNN dash detector (Local CPU/GPU)
===========================================================
Load the per-frame features + labels from Stage 2, train the DNN with a
per-frame heatmap loss, log metrics, and save a checkpoint.

Examples:
    python -m dash_code.train
    python -m dash_code.train --epochs 40 --hidden 128
"""

import argparse
import json
from pathlib import Path

# Allow running this file directly (`python dash_code/train.py`) as well as
# as a module (`python -m dash_code.train`). When run directly there is no
# package context, so put the repo root on sys.path for `from helper... import`.
if __package__ in (None, ""):
    import os
    import sys
    _here = os.path.abspath(__file__)
    sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as C
from . import dataset as ds
from . import peaks as pk
from .dnn import DNN, heatmap_loss


def val_loss_over(model, loader, device):
    """Mean heatmap loss over the (masked) val windows — a comparable val number."""
    model.eval()
    s = n = 0
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            s += float(heatmap_loss(model(x), y, mask=m,
                                    pos_weight=C.HEATMAP_POS_WEIGHT))
            n += 1
    return s / max(n, 1)


def video_probs(model, arrays, device):
    """Per FULL video: (prob_track, gt_completion_centers). Probs are computed
    once here so a threshold sweep can reuse them. Whole videos (not windows) so
    peaks aren't split or double-counted at window edges."""
    model.eval()
    out = []
    with torch.no_grad():
        for feat, lab in arrays:
            x = torch.from_numpy(feat[None]).to(device)             # [1, T, F]
            prob = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()   # [T]
            out.append((prob, pk.gt_centers_from_heatmap(lab)))
    return out


def peak_eval(video_probs_list, threshold, tol, min_dist):
    """Count MAE + peak-timing P/R/F1 at one NMS threshold (reuses cached probs)."""
    preds = [pk.nms_peaks(prob, threshold, min_dist) for prob, _ in video_probs_list]
    gts   = [g for _, g in video_probs_list]
    return pk.count_metrics(preds, gts, tol)


def sweep_peaks(video_probs_list, thresholds, tol, min_dist):
    """Peak metrics across NMS thresholds. Returns one row per threshold."""
    rows = []
    for t in thresholds:
        m = peak_eval(video_probs_list, t, tol, min_dist)
        rows.append(dict(threshold=round(float(t), 2), **m))
    return rows


def main():
    import sys
    sys.stdout = open("train_log.txt", "w", encoding="utf-8")
    ap = argparse.ArgumentParser(description="Phase 3: train the DNN dash detector.")
    ap.add_argument("--epochs", type=int, default=C.N_EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--hidden", type=int, default=C.HIDDEN)
    ap.add_argument("--layers", type=int, default=4,
                    help="number of stacked delay layers")
    ap.add_argument("--window", type=int, default=C.WINDOW)
    ap.add_argument("--stride", type=int, default=C.STRIDE)
    ap.add_argument("--loss", choices=["heatmap", "bce", "focal"], default="heatmap",
                    help="completion target is soft -> heatmap regression; bce/focal "
                         "accepted for back-compat but routed to heatmap")
    ap.add_argument("--threshold", type=float, default=C.PEAK_THRESHOLD,
                    help="NMS confidence cutoff used during training (final value swept)")
    ap.add_argument("--peak-tol", type=int, default=C.PEAK_MATCH_TOL,
                    help="frame tolerance when matching a predicted peak to a GT completion")
    ap.add_argument("--min-dist", type=int, default=C.PEAK_MIN_DIST,
                    help="minimum frames between two counted peaks (1-D NMS)")
    ap.add_argument("--val-frac", type=float, default=C.VAL_FRAC)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out", type=str, default=str(C.CHECKPOINT_DIR))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    files = ds.list_processed()
    if not files:
        print(f"No processed .npz in {C.FEATURES_DIR}. Run Phase 1 "
              f"(`process_data.py --stage features`) first.")
        return
    train_files, val_files = ds.split_videos(files, args.val_frac, args.seed)
    print(f"videos: {len(files)} total  ->  {len(train_files)} train / {len(val_files)} val")

    train_ds = ds.WindowDataset(train_files, args.window, args.stride)
    val_ds   = ds.WindowDataset(val_files,   args.window, args.stride)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=False)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
    print(f"windows: {len(train_ds)} train / {len(val_ds)} val  (window={args.window})")

    in_dim = train_ds.arrays[0][0].shape[1]
    n_peaks = sum(len(pk.gt_centers_from_heatmap(lab)) for _, lab in train_ds.arrays)
    if args.loss != "heatmap":
        print(f"[note] --loss {args.loss} ignored: the completion target is a soft "
              f"heatmap, training with heatmap regression (weighted MSE).")
    print(f"in_dim={in_dim}  train completion peaks={n_peaks}  "
          f"loss=heatmap (pos_weight={C.HEATMAP_POS_WEIGHT})")

    model = DNN(in_dim=in_dim, hidden=args.hidden, layers=args.layers).to(device)
    print(f"model: {args.layers} fixed delay layer(s), hidden={args.hidden}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def loss_fn(logits, y, m):
        return heatmap_loss(logits, y, mask=m, pos_weight=C.HEATMAP_POS_WEIGHT)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_f1 = [], (-1e9, 0.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = nb = 0
        layer_gn = [0.0] * len(model.layers)
        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x)
            loss = loss_fn(logits, y, m)
            opt.zero_grad(); loss.backward()
            # per-layer gradient L2 norm — watch the EARLY layers' grads shrink as
            # you add depth (and recover once residual/norm are on).
            for li, lyr in enumerate(model.layers):
                sq = sum(p.grad.pow(2).sum().item() for p in lyr.parameters()
                         if p.grad is not None)
                layer_gn[li] += sq ** 0.5
            opt.step()
            run_loss += loss.item(); nb += 1
        grad_norms = [round(g / max(nb, 1), 6) for g in layer_gn]

        # peak-based val: count MAE + timing F1 over whole videos (the numbers the
        # parent counter cares about), plus a comparable val loss.
        val = {}
        if len(val_ds):
            vloss = val_loss_over(model, val_ld, device)
            vp = video_probs(model, val_ds.arrays, device)
            m = peak_eval(vp, args.threshold, args.peak_tol, args.min_dist)
            val = dict(loss=vloss, count_mae=m["count_mae"], f1=m["f1"],
                       precision=m["precision"], recall=m["recall"])
        tr_loss = run_loss / max(nb, 1)
        history.append(dict(epoch=epoch, train_loss=tr_loss, grad_norms=grad_norms, **val))
        msg = f"epoch {epoch:3d}  train_loss {tr_loss:.4f}"
        if val:
            msg += (f"  val_loss {val['loss']:.4f}  cntMAE {val['count_mae']:.3f}  "
                    f"P {val['precision']:.3f}  R {val['recall']:.3f}  F1 {val['f1']:.3f}")
        msg += "  | grad " + " ".join(f"L{i+1} {g:.1e}" for i, g in enumerate(grad_norms))
        print(msg)

        # select by timing F1 (tie-break: lower count MAE); no val -> train loss
        score = (val["f1"], -val["count_mae"]) if val else (tr_loss * -1, 0.0)
        if score > best_f1:
            best_f1 = score
            torch.save(dict(
                model_state=model.state_dict(),
                in_dim=in_dim, hidden=args.hidden,
                layers=args.layers,
                feat_dim=C.FEAT_DIM, img_size=C.IMG_SIZE,
                backbone=str(C.BACKBONE_CKPT) or "imagenet_resnet18", fps=C.FPS,
                threshold=args.threshold, peak_min_dist=args.min_dist,
                peak_match_tol=args.peak_tol, sigma=C.GAUSS_SIGMA,
                completion_ms=C.COMPLETION_MS, epoch=epoch, val=val,
            ), out_dir / "sdsnn.pt")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nbest val timing-F1 (@thr {args.threshold}): {best_f1[0]:.3f}"
          if len(val_ds) else "\n(no val set)")

    # --- NMS-threshold sweep (peak counting) on the best model -----------------
    ckpt_path = out_dir / "sdsnn.pt"
    if len(val_ds) and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        vp = video_probs(model, val_ds.arrays, device)
        thresholds = [round(float(t), 2) for t in np.arange(0.05, 0.96, 0.05)]
        rows = sweep_peaks(vp, thresholds, args.peak_tol, args.min_dist)

        print(f"\nNMS-threshold sweep on val  (match tol={args.peak_tol}f, "
              f"min-dist={args.min_dist}f):")
        print("  thr | count MAE | timing P/R/F1      | tp/fp/fn")
        for r in rows:
            print(f"  {r['threshold']:.2f} | {r['count_mae']:>8.3f}  | "
                  f"{r['precision']:.2f}/{r['recall']:.2f}/{r['f1']:.2f} | "
                  f"{r['tp']}/{r['fp']}/{r['fn']}")

        # pick the threshold with the best timing F1 (tie-break: lower count MAE)
        best = max(rows, key=lambda r: (r["f1"], -r["count_mae"]))
        print(f"\nbest timing F1={best['f1']:.3f} @ thr={best['threshold']:.2f}  "
              f"(count MAE={best['count_mae']:.3f}  P={best['precision']:.2f}  "
              f"R={best['recall']:.2f})")

        # bake the chosen operating threshold into the checkpoint for stage4.py
        ck = torch.load(ckpt_path, map_location=device)
        ck["threshold"]      = best["threshold"]
        ck["peak_val"]       = best
        ck["peak_match_tol"] = args.peak_tol
        ck["peak_min_dist"]  = args.min_dist
        torch.save(ck, ckpt_path)

        with open(out_dir / "sweep.json", "w") as f:
            json.dump(dict(sweep=rows, best=best,
                           peak_match_tol=args.peak_tol,
                           peak_min_dist=args.min_dist), f, indent=2)
        print(f"sweep      -> {out_dir / 'sweep.json'}")

    print(f"checkpoint -> {ckpt_path}")
    print(f"metrics    -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
