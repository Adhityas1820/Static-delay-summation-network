"""
stage4.py  —  STAGE 4 (local): COUNT dashes with the trained SDSNN, score the count
====================================================================================
Loads the trained checkpoint, runs the SAME pipeline as training — frozen
ResNet18 per-frame features (the spatial front-end) -> SDSNN temporal head ->
per-frame COMPLETION probability (a soft bump peaking when a dash has fully
unfolded) — then turns that probability track into a dash COUNT with 1-D
non-max suppression (NMS): take the highest frame above the threshold, count it
as one dash, suppress everything within min-dist, repeat. #peaks == #dashes.

This is the count the parent Marvel Rivals counter actually wants. NMS (not
run-length grouping) is what splits CHAINED dashes whose completion bumps sit a
few frames apart.

Quality is the COUNT accuracy vs. ground truth:
    count MAE   = mean |pred_count - gt_count|   over videos
    timing F1   = peaks matched to GT completions within +/- match-tol frames
Ground truth completions come from the dash_counter heuristic per video, or
from a reviewed CSV's completion_frame column (--labels).

Output: predictions/counts_per_video.csv — video, pred_count, gt_count,
count_err, tp, fp, fn. Annotated review videos (red flash = predicted
completion, green = ground-truth completion, live dash tally) render by default
to predictions/outputvideos/. Pass --no-overlay to write only the CSV.

Usage:
    python stage4.py                                   # all clips in test dir
    python stage4.py --dir path/to/clips --labels path/to/dash_intervals.csv
    python stage4.py --no-overlay
    python stage4.py --ckpt path/to/sdsnn.pt --threshold 0.6 --min-dist 12
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

import json
from dash_code import config as C
from dash_code import features as feat
from dash_code import labels as lab
from dash_code import io_utils
from dash_code.dash_counter import open_video, count_dashes
from dash_code import peaks as pk
from dash_code.dnn import DNN

# --- hardcoded local defaults (override with --dir / --ckpt / --labels) -----
TEST_DIR  = str(C.PROJECT_ROOT / "test set")
CKPT_PATH = str(C.PROJECT_ROOT / "checkpoints" / "sdsnn.pt")


# ---------------------------------------------------------------------------
# model + per-frame prediction
# ---------------------------------------------------------------------------
def load_model(ckpt_path: Path, device):
    ck = torch.load(str(ckpt_path), map_location=device)
    # rebuild the SAME architecture the checkpoint was trained with.
    model = DNN(in_dim=ck.get("in_dim", C.FEAT_DIM),
                  hidden=ck.get("hidden", 64),
                  layers=ck.get("layers", 4)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def predict_mask(model, features, device, threshold):
    """features [T, F] -> (prob [T], dash mask [T] bool). Two states only."""
    if features.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)
    x = torch.from_numpy(features[None]).to(device)            # [1, T, F]
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()  # [T]
    return prob, prob >= threshold


# ---------------------------------------------------------------------------
# ground truth + IoU
# ---------------------------------------------------------------------------
def load_truth(labels_path: Path) -> dict:
    """Reviewed labels CSV -> {video_name: [(start_frame, end_frame), ...]}.

    Reads the same format dash_intervals.csv uses (the columns we need are
    'video', 'start_frame', 'end_frame'). Missing file -> empty mapping.
    """
    truth = defaultdict(list)
    if not labels_path.exists():
        return truth
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = int(float(row["start_frame"]))
                e = int(float(row["end_frame"]))
            except (KeyError, ValueError, TypeError):
                continue
            truth[row["video"]].append((s, e))
    return truth


def _probe_fps(video_path) -> float:
    """Source fps, needed to turn the detector's dash-start SECONDS into frames."""
    cap, tmp = open_video(video_path)
    fps = (cap.get(cv2.CAP_PROP_FPS) or C.FPS) if cap.isOpened() else C.FPS
    cap.release()
    if tmp:
        Path(tmp).unlink(missing_ok=True)
    return float(fps)


def truth_from_detector(video_path, n_frames):
    """Ground truth straight from the dash_counter heuristic, built EXACTLY like
    Stage 1: detected dash-start seconds -> frame indices (round(sec * fps)) ->
    [start-PRE_FRAMES, start+POST_FRAMES] intervals -> per-frame mask, clamped to
    n_frames (the model's decoded length, so it lines up with the prediction).

    Returns (gt_mask bool [n_frames], n_dash_events). Caveat: this measures how
    faithfully the SDSNN reproduces the heuristic detector it was trained on, not
    absolute dash-detection accuracy.
    """
    res = count_dashes(Path(video_path))          # (name, total, ts, combos, dash_secs)
    dash_secs = res[4]
    fps = _probe_fps(video_path)
    dash_frames = [int(round(s * fps)) for s in dash_secs]
    intervals = [lab.build_interval(f, n_frames) for f in dash_frames]
    return lab.labels_from_intervals(intervals, n_frames).astype(bool), len(dash_frames)


def truth_centers_from_detector(video_path, n_frames):
    """Ground-truth COMPLETION centers from the dash_counter heuristic: each
    detected dash-start -> its completion frame (the spike center the SDSNN is
    trained to fire at). Returns (sorted centers, n_dash_events)."""
    res = count_dashes(Path(video_path))
    dash_secs = res[4]
    fps = _probe_fps(video_path)
    dash_frames = [int(round(s * fps)) for s in dash_secs]
    centers = sorted(lab.completion_frame(f, n_frames) for f in dash_frames)
    return centers, len(dash_frames)


def iou_score(pred_mask, gt_mask) -> float:
    """IoU of two per-frame boolean masks. Both empty -> 1.0 (full agreement)."""
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    if union == 0:
        return 1.0
    return inter / union


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def write_counts_csv(out_dir: Path, rows: list,
                     name: str = "counts_per_video.csv") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fields = ["video", "pred_count", "gt_count", "count_err", "tp", "fp", "fn"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def render_overlay(video_path: Path, pred_peaks, gt_centers, prob, out_dir: Path,
                   threshold: float, flash=3) -> Path:
    """Annotate a video with the COMPLETION peaks: a red top band flashes for a
    few frames around each PREDICTED completion (a counted dash), a green bottom
    band around each GROUND-TRUTH completion. A live 'dashes: k/N' tally and the
    per-frame probability stay on so you can scrub and cross-check the count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(prob)
    pred_set = set(pred_peaks)
    gt_set   = set(gt_centers)

    def near(idx, centers):
        return any(abs(idx - c) <= flash for c in centers)

    cap, tmp = open_video(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or C.FPS
    w, h = C.REVIEW_W, C.REVIEW_H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = out_dir / f"{video_path.stem}_overlay.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    bar_w, pad = 46, 12
    x0 = w - bar_w - pad
    top, bot = 14, h - 14
    span = bot - top

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (w, h))

        if near(idx, pred_set):                       # red top band = PRED completion
            cv2.rectangle(frame, (0, 0), (w - 1, 14), (0, 0, 255), -1)
        if near(idx, gt_set):                         # green bottom band = GT completion
            cv2.rectangle(frame, (0, h - 15), (w - 1, h - 1), (0, 255, 0), -1)
        if idx in pred_set:
            cv2.putText(frame, "DASH!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 0, 255), 3)

        seen = sum(1 for p in pred_peaks if p <= idx)
        p = prob[idx] if idx < n else 0.0
        
        # right-edge probability gauge + threshold reference line
        cv2.rectangle(frame, (x0, top), (x0 + bar_w, bot), (60, 60, 60), 1)
        fill = int(span * max(0.0, min(1.0, p)))
        col = (0, 255, 0) if p >= threshold else (0, 165, 255)
        cv2.rectangle(frame, (x0, bot - fill), (x0 + bar_w, bot), col, -1)
        ty = bot - int(span * threshold)
        cv2.line(frame, (x0 - 4, ty), (x0 + bar_w + 4, ty), (0, 0, 255), 1)
        cv2.putText(frame, f"{p:.2f}", (x0 - 6, top - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        ts = lab.fmt_ts(idx * C.MS_PER_FRAME)
        cv2.putText(frame, f"dashes: {seen}/{len(pred_peaks)}", (30, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"f{idx}  {ts}  p={p:.2f}", (30, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
        idx += 1

    writer.release()
    cap.release()
    if tmp:
        Path(tmp).unlink(missing_ok=True)
    return out_path


# ---------------------------------------------------------------------------
def list_videos(d: Path) -> list:
    if not d.exists():
        return []
    return sorted(f for f in d.iterdir()
                  if f.is_file() and f.suffix.lower() in C.VIDEO_EXTENSIONS)


def main():
    import sys
    sys.stdout = open("output.txt", "w", encoding="utf-8")
    ap = argparse.ArgumentParser(
        description="Stage 4: count dashes with the SDSNN completion detector "
                    "(1-D NMS over the per-frame completion probability) and "
                    "score the count vs. ground truth.")
    ap.add_argument("--video", help="single video file to run on")
    ap.add_argument("--dir", help="folder of videos (default: TEST_DIR)")
    ap.add_argument("--labels", default=None,
                    help="use this reviewed labels CSV as ground truth (its "
                         "completion_frame column) instead of the dash_counter heuristic")
    ap.add_argument("--ckpt", default=CKPT_PATH,
                    help="trained checkpoint (default: claudeCheck/checkpoints/sdsnn.pt)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the checkpoint's tuned NMS threshold")
    ap.add_argument("--min-dist", type=int, default=None,
                    help="override min frames between counted peaks (NMS)")
    ap.add_argument("--out", default=str(C.PROJECT_ROOT / "predictions"),
                    help="output folder for the CSV / overlays")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip the annotated review videos (predictions/outputvideos/)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}. Train it (Stage 3) or pass --ckpt.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(ckpt_path, device)
    threshold = args.threshold if args.threshold is not None else ck.get("threshold", C.PEAK_THRESHOLD)
    min_dist  = args.min_dist if args.min_dist is not None else ck.get("peak_min_dist", C.PEAK_MIN_DIST)
    tol       = ck.get("peak_match_tol", C.PEAK_MATCH_TOL)
    print(f"device: {device}  |  backbone: {ck.get('backbone')}  "
          f"|  arch: {ck.get('layers', 2)}L h{ck.get('hidden')} "
          f"res={ck.get('residual', False)} norm={ck.get('norm', False)}  "
          f"|  thr: {threshold}  min-dist: {min_dist}f  match-tol: {tol}f")

    if args.video:
        videos = [Path(args.video)]
    else:
        vdir = Path(args.dir) if args.dir else Path(TEST_DIR)
        print(f"videos dir: {vdir}")
        videos = list_videos(vdir)
    if not videos:
        print("No videos to run on. Pass --video or --dir.")
        return

    # Ground truth completion centers: by default the dash_counter heuristic per
    # video; pass --labels to use a reviewed CSV's completion_frame column.
    use_csv = args.labels is not None
    gt_centers_csv = io_utils.read_completion_centers(Path(args.labels)) if use_csv else {}
    if use_csv:
        print(f"ground truth: CSV {args.labels}  ({len(gt_centers_csv)} labeled video(s))")
    else:
        print("ground truth: dash_counter heuristic (detected per video)")

    out_dir = Path(args.out)
    overlay_dir = out_dir / "outputvideos"
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pred_lists, gt_lists = [], []     # for aggregate count MAE + timing F1
    for v in videos:
        if not v.exists():
            print(f"  [{v.name}] not found — skipping")
            continue
            
        feat_cache_path = cache_dir / f"{v.stem}_features.npy"
        if feat_cache_path.exists():
            features = np.load(feat_cache_path)
        else:
            features = feat.extract_cnn_features(v)
            np.save(feat_cache_path, features)
            
        prob, _ = predict_mask(model, features, device, 0.0)   # keep prob; NMS counts
        n = len(prob)
        if n == 0:
            print(f"  [{v.name}] could not decode (0 frames) — file may be corrupt "
                  f"or an undecodable codec. Left blank.")
            rows.append(dict(video=v.name, pred_count="", gt_count="",
                             count_err="", tp="", fp="", fn=""))
            continue

        pred_peaks = pk.nms_peaks(prob, threshold, min_dist)

        # ground-truth completion centers (CSV column or detector heuristic)
        if use_csv:
            if v.name not in gt_centers_csv:
                print(f"  [{v.name}] no GT row in {Path(args.labels).name} — "
                      f"pred {len(pred_peaks)} dashes, not scored")
                rows.append(dict(video=v.name, pred_count=len(pred_peaks),
                                 gt_count="", count_err="", tp="", fp="", fn=""))
                if not args.no_overlay:
                    render_overlay(v, pred_peaks, [], prob, overlay_dir, threshold)
                continue
            gt = sorted(c for c in gt_centers_csv[v.name] if 0 <= c < n)
        else:
            gt_cache_path = cache_dir / f"{v.stem}_gt.json"
            if gt_cache_path.exists():
                with open(gt_cache_path, "r") as f:
                    gt = json.load(f)
            else:
                gt, _ = truth_centers_from_detector(v, n)
                with open(gt_cache_path, "w") as f:
                    json.dump(gt, f)

        tp, fp, fn, _ = pk.match_peaks(pred_peaks, gt, tol)
        pred_lists.append(pred_peaks); gt_lists.append(gt)
        rows.append(dict(video=v.name, pred_count=len(pred_peaks), gt_count=len(gt),
                         count_err=len(pred_peaks) - len(gt), tp=tp, fp=fp, fn=fn))
        print(f"  [{v.name}] pred {len(pred_peaks)} / gt {len(gt)} dashes  "
              f"(err {len(pred_peaks)-len(gt):+d}; tp {tp} fp {fp} fn {fn})")

        if not args.no_overlay:
            op = render_overlay(v, pred_peaks, gt, prob, overlay_dir, threshold)
            if op:
                print(f"      overlay -> {op}")

    csv_path = write_counts_csv(out_dir, rows)
    agg = pk.count_metrics(pred_lists, gt_lists, tol) if pred_lists else None

    print("\n" + "=" * 64)
    print("DASH COUNT PER VIDEO  (pred / gt)")
    print("=" * 64)
    width = max((len(r["video"]) for r in rows), default=10)
    for r in rows:
        if r["gt_count"] == "":
            tail = f"pred {r['pred_count']}" if r["pred_count"] != "" else "(no decode)"
        else:
            tail = f"{r['pred_count']:>3} / {r['gt_count']:<3}  (err {r['count_err']:+d})"
        print(f"  {r['video']:<{width}}  {tail}")
    print("-" * 64)
    if agg:
        print(f"  count MAE {agg['count_mae']:.3f}  |  timing F1 {agg['f1']:.3f} "
              f"(P {agg['precision']:.2f} R {agg['recall']:.2f})  over "
              f"{len(pred_lists)} scored video(s)")
    print(f"\nCSV -> {csv_path}")


if __name__ == "__main__":
    main()
