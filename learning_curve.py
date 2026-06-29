"""
learning_curve.py — does adding training data help? (held-out val FIXED)
========================================================================
Trains the SDSNN head on 25/50/75/100% of the TRAINING clips while keeping the
SAME validation clips every time, so the only thing that changes is how much
data the model learns from. Nested subsets (25% is inside 50% is inside ...) so
each step only ADDS clips. Reuses train.py's evaluate() + sweep_full() so the
event-F1 is comparable to a normal run.

Read the SHAPE: if event-F1 still climbs at 100%, more data helps. If it has
flattened, data volume is not the current bottleneck. (Caveat: the val set is
tiny ~19 events, so treat points as noisy and read the trend, not the decimals.)

    python learning_curve.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from dash_code import config as C
from dash_code import dataset as ds
from dash_code.sdsnn import SDSNN, weighted_bce
from dash_code.train import evaluate, sweep_full

EPOCHS, FRACTIONS, SEED = 50, [0.25, 0.5, 0.75, 1.0], C.SEED


def count_events(arrays):
    """contiguous runs of label==1 = dash events, across a set of clips."""
    n = 0
    for _, lab in arrays:
        prev = 0
        for v in (lab > 0.5).astype(int):
            if v and not prev:
                n += 1
            prev = v
    return n


def train_once(train_files, val_ds, device):
    train_ds = ds.WindowDataset(train_files, C.WINDOW, C.STRIDE)
    train_ld = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=C.BATCH_SIZE, shuffle=False)

    in_dim = train_ds.arrays[0][0].shape[1]
    pos, neg = train_ds.pos_neg_counts()
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = SDSNN(in_dim=in_dim, hidden=C.HIDDEN, max_delay=C.MAX_DELAY, layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)

    best_f1, best_state = -1.0, None
    for _ in range(EPOCHS):
        model.train()
        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = weighted_bce(model(x), y, pos_weight, mask=m)
            opt.zero_grad(); loss.backward(); opt.step()
        v = evaluate(model, val_ld, device, 0.5)
        if v["f1"] > best_f1:
            best_f1, best_state = v["f1"], {k: t.clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    rows = sweep_full(model, val_ds.arrays, device,
                      [round(float(t), 2) for t in np.arange(0.05, 0.96, 0.05)], 5, 3)
    return max(rows, key=lambda r: r["event_f1"]), len(train_ds), count_events(train_ds.arrays)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = ds.list_processed()
    train_files, val_files = ds.split_videos(files, C.VAL_FRAC, SEED)
    val_ds = ds.WindowDataset(val_files, C.WINDOW, C.STRIDE)
    val_events = count_events(val_ds.arrays)
    print(f"clips: {len(files)} -> {len(train_files)} train / {len(val_files)} val "
          f"(val has {val_events} dash events — FIXED across all points)\n")

    # one shuffle, then nested prefixes so each step only adds clips
    rng = np.random.default_rng(SEED)
    order = list(train_files)
    rng.shuffle(order)

    print(f"{'frac':>5} {'clips':>6} {'tr_events':>10} {'best_eF1':>9} {'thr':>5} {'P':>5} {'R':>5}")
    results = []
    for frac in FRACTIONS:
        k = max(1, int(round(frac * len(order))))
        subset = order[:k]
        best, nwin, ev = train_once(subset, val_ds, device)
        results.append((frac, k, ev, best["event_f1"]))
        print(f"{frac:>5.2f} {k:>6} {ev:>10} {best['event_f1']:>9.3f} "
              f"{best['threshold']:>5.2f} {best['event_p']:>5.2f} {best['event_r']:>5.2f}")

    print("\nshape:")
    for i, (frac, k, ev, f1) in enumerate(results):
        bar = "#" * int(round(f1 * 40))
        delta = "" if i == 0 else f"  ({f1-results[i-1][3]:+.3f})"
        print(f"  {int(frac*100):>3}% {f1:.3f} |{bar}{delta}")
    print("\nStill climbing at 100% -> more data likely helps. "
          "Flat -> volume isn't the current limit.")


if __name__ == "__main__":
    main()
