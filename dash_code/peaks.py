"""
peaks.py
--------
Turn the SDSNN's per-frame completion-probability track into discrete dash
COUNTS, and score them. Shared by Stage 3 (train.py eval) and Stage 4
(stage4.py counting) so training and inference decode peaks identically.

The completion reframe makes #peaks == #dashes: the model emits a smooth bump
at each dash's falling edge, and 1-D non-max suppression collapses each bump to
one counted dash. NMS (not run-length grouping) is what separates CHAINED dashes
whose bumps sit a few frames apart — a plateau-threshold rule would merge them.
"""

import numpy as np

from . import config as C


def nms_peaks(prob, threshold=None, min_dist=None):
    """1-D non-max suppression. Greedily take the highest frame >= threshold,
    record it, suppress everything within +/- min_dist, repeat. Returns sorted
    peak frame indices (= the counted dashes)."""
    threshold = C.PEAK_THRESHOLD if threshold is None else threshold
    min_dist  = C.PEAK_MIN_DIST  if min_dist  is None else min_dist
    prob = np.asarray(prob)
    cand = np.where(prob >= threshold)[0]
    if cand.size == 0:
        return []
    order = cand[np.argsort(-prob[cand])]
    taken = np.zeros(len(prob), dtype=bool)
    chosen = []
    for i in order:
        lo, hi = max(0, i - min_dist), min(len(prob), i + min_dist + 1)
        if taken[lo:hi].any():
            continue
        chosen.append(int(i)); taken[i] = True
    return sorted(chosen)


def gt_centers_from_heatmap(y, min_dist=None):
    """Recover the completion centers from a soft GT heatmap (peaks at ~1.0).
    Uses the same NMS at a high cutoff so it returns exactly one center per bump
    even where chained bumps overlap."""
    return nms_peaks(y, threshold=0.5, min_dist=min_dist)


def match_peaks(pred, gt, tol=None):
    """Greedy nearest 1-to-1 match within tol frames. Returns (tp, fp, fn,
    timing_errors) where timing_errors are |pred-gt| for matched pairs."""
    tol = C.PEAK_MATCH_TOL if tol is None else tol
    used = [False] * len(gt)
    tp, errs = 0, []
    for p in sorted(pred):
        best, bd = -1, tol + 1
        for j, g in enumerate(gt):
            if used[j]:
                continue
            d = abs(p - g)
            if d <= tol and d < bd:
                bd, best = d, j
        if best >= 0:
            used[best] = True; tp += 1; errs.append(bd)
    return tp, len(pred) - tp, len(gt) - tp, errs


def count_metrics(pred_list, gt_list, tol=None):
    """Aggregate over videos: count MAE (mean |pred_count - gt_count|) and micro
    peak-timing precision/recall/F1 + mean timing error. pred_list/gt_list are
    lists of peak-index lists (one per video)."""
    abs_err, TP, FP, FN, all_err = [], 0, 0, 0, []
    for pred, gt in zip(pred_list, gt_list):
        tp, fp, fn, errs = match_peaks(pred, gt, tol)
        abs_err.append(abs(len(pred) - len(gt)))
        TP += tp; FP += fp; FN += fn; all_err += errs
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec  = TP / (TP + FN) if (TP + FN) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(count_mae=float(np.mean(abs_err)) if abs_err else 0.0,
                precision=prec, recall=rec, f1=f1,
                mean_timing_err=float(np.mean(all_err)) if all_err else None,
                tp=TP, fp=FP, fn=FN)
