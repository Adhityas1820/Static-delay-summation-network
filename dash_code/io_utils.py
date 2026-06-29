"""
io_utils.py
-----------
Reading/writing the artefacts that pass between pipeline stages:

  * dash_intervals.csv  — the human-review label file (also the source of
    truth for which intervals become training labels; edit it before the
    feature stage if a detection is wrong).
  * processed/features/<stem>.npz — per-video feature + label arrays.
  * processed/manifest.csv — index of saved videos (frames, positives).
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config as C

CSV_FIELDS = ["video",
              "dash_start_ms", "dash_start_ts",
              "interval_start_ms", "interval_start_ts",
              "interval_end_ms", "interval_end_ts",
              "start_frame", "end_frame",
              "completion_frame", "completion_ms", "completion_ts"]


# --- intervals CSV (human-reviewable labels) -------------------------------
def write_intervals_csv(rows: list, path: Path = None):
    path = Path(path or C.LABELS_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def read_intervals_csv(path: Path = None) -> dict:
    """
    Returns {video_name: [(start_frame, end_frame), ...]} from the CSV, so a
    human-corrected file is what drives the labels.
    """
    path = Path(path or C.LABELS_CSV)
    by_video = defaultdict(list)
    if not path.exists():
        return by_video
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_video[r["video"]].append(
                (int(r["start_frame"]), int(r["end_frame"]))
            )
    return by_video


def read_completion_centers(path: Path = None) -> dict:
    """
    Returns {video_name: [completion_frame, ...]} — the per-dash spike centers
    that drive the Gaussian training target (Stage 2). Reads the completion_frame
    column; falls back to a pre-completion CSV's end_frame (the falling edge) so
    an old reviewed file still works.
    """
    path = Path(path or C.LABELS_CSV)
    by_video = defaultdict(list)
    if not path.exists():
        return by_video
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            val = r.get("completion_frame")
            c = int(float(val)) if val not in (None, "") else int(float(r["end_frame"]))
            by_video[r["video"]].append(c)
    return by_video


# --- processed feature/label arrays ----------------------------------------
def save_processed(stem: str, features: np.ndarray, labels: np.ndarray,
                   src_fps: float, feature_names: list):
    C.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = C.FEATURES_DIR / f"{stem}.npz"
    np.savez_compressed(
        out,
        features=features.astype(np.float32),
        labels=labels.astype(np.float32),
        src_fps=np.float32(src_fps),
        feature_names=np.array(feature_names),
    )
    return out


def write_manifest(records: list, path: Path = None):
    """records: list of dicts with video, n_frames, n_pos, n_features."""
    path = Path(path or (C.PROCESSED_DIR / "manifest.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video", "n_frames", "n_peaks", "n_pos", "pos_frac", "n_features"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(r)
    return path
