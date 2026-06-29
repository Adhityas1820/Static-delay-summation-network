"""
labels.py
---------
Turn dash START points into (a) human-reviewable intervals and (b) per-frame
binary labels for SDSNN training.

Interval geometry (from config): start = point - 50 ms (3 frames),
end = point + 430 ms (26 frames). One frame is labelled 1 iff it falls inside
any dash interval, else 0 — this is exactly the per-frame target the SDSNN's
weighted BCE / focal loss is trained against.
"""

import numpy as np

from . import config as C

# interval offsets in frames (rounded from the ms prior)
PRE_FRAMES  = int(round(C.PRE_DASH_MS  / C.MS_PER_FRAME))   # 3
POST_FRAMES = int(round(C.POST_DASH_MS / C.MS_PER_FRAME))   # 26

# completion offset: how many frames after the detected START the dash is fully
# unfolded (the spike center). The [start-PRE, start+POST] interval above is kept
# only for the human review WINDOW (overlays); the TRAINING target is a soft
# Gaussian peaking COMPLETION_FRAMES after the start. See config / decisions.
COMPLETION_FRAMES = int(round(C.COMPLETION_MS / C.MS_PER_FRAME))   # 27


def fmt_ts(ms: float) -> str:
    """Milliseconds -> 'M:SS:mmm' (minutes:seconds:milliseconds) for review."""
    ms    = int(round(ms))
    mins  = ms // 60000
    secs  = (ms % 60000) // 1000
    rem   = ms % 1000
    return f"{mins}:{secs:02d}:{rem:03d}"


def build_interval(dash_frame: int, n_frames: int) -> tuple:
    """Clamp [dash_frame - PRE, dash_frame + POST] to [0, n_frames - 1]."""
    start = max(0, dash_frame - PRE_FRAMES)
    end   = min(n_frames - 1, dash_frame + POST_FRAMES)
    return start, end


def completion_frame(dash_frame: int, n_frames: int) -> int:
    """The frame the dash is fully unfolded (the spike center), clamped to the
    clip. This is what the Gaussian completion target peaks at."""
    return min(n_frames - 1, max(0, dash_frame + COMPLETION_FRAMES))


def interval_csv_rows(detection: dict) -> list:
    """
    Build the human-review rows for one video's detections.

    Columns: video, dash_start_ms, interval_start_ms, interval_end_ms,
             start_frame, end_frame
    """
    rows = []
    n_frames = detection["n_frames"]
    for f in detection["dash_frames"]:
        start_f, end_f = build_interval(f, n_frames)
        comp_f   = completion_frame(f, n_frames)
        dash_ms  = round(f * C.MS_PER_FRAME, 1)
        start_ms = round(start_f * C.MS_PER_FRAME, 1)
        end_ms   = round(end_f * C.MS_PER_FRAME, 1)
        comp_ms  = round(comp_f * C.MS_PER_FRAME, 1)
        rows.append(dict(
            video             = detection["video"],
            dash_start_ms     = dash_ms,
            dash_start_ts     = fmt_ts(dash_ms),
            interval_start_ms = start_ms,
            interval_start_ts = fmt_ts(start_ms),
            interval_end_ms   = end_ms,
            interval_end_ts   = fmt_ts(end_ms),
            start_frame       = start_f,
            end_frame         = end_f,
            completion_frame  = comp_f,
            completion_ms     = comp_ms,
            completion_ts     = fmt_ts(comp_ms),
        ))
    return rows


def labels_from_intervals(intervals: list, n_frames: int) -> np.ndarray:
    """
    intervals: list of (start_frame, end_frame) INCLUSIVE.
    Returns float32 array of shape [n_frames] with 1.0 inside any interval.

    This is the OLD per-frame "dash vs non-dash" target. The pipeline now trains
    on gaussian_labels() instead (the completion-spike reframe); this is kept for
    the review overlays and any A/B against the old target.
    """
    y = np.zeros(n_frames, dtype=np.float32)
    for start_f, end_f in intervals:
        s = max(0, int(start_f))
        e = min(n_frames - 1, int(end_f))
        if e >= s:
            y[s:e + 1] = 1.0
    return y


def gaussian_labels(centers: list, n_frames: int, sigma: float = None) -> np.ndarray:
    """Soft completion target [n_frames] in [0, 1]: the elementwise MAX of a
    Gaussian bump centred at each completion frame. #peaks == #dashes.

    centers: completion frame indices (one per dash). A single hot frame won't
    train at <1% positives, so each dash becomes a smooth bump (peak 1.0 at the
    completion frame, fading over ~sigma frames); 1-D NMS later collapses each
    bump back to one counted dash. MAX (not sum) keeps overlapping chained-dash
    bumps capped at 1.0 while still leaving a separable peak per dash.
    """
    sigma = C.GAUSS_SIGMA if sigma is None else sigma
    y = np.zeros(n_frames, dtype=np.float32)
    if n_frames == 0:
        return y
    t = np.arange(n_frames)
    two_s2 = 2.0 * sigma * sigma
    for c in centers:
        c = int(c)
        if 0 <= c < n_frames:
            y = np.maximum(y, np.exp(-((t - c) ** 2) / two_s2))
    return y.astype(np.float32)
