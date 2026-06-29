"""
dash_counter.py
---------------
Counts dashes in Marvel Rivals gameplay clips using contour-based white
detection + zoom exclusion.

Usage:
    python dash_counter.py
"""

import json
import re
import shutil
import tempfile
import multiprocessing as mp
import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR  = "unsorted_videos"
OUTPUT_DIR = "dash_counts"

PROCESS_FPS = 60

SLOT2_SEARCH = (1575, 1625, 965, 1000)
SLOT3_SEARCH = (1500, 1550, 965, 1000)
SLOT2_LABEL  = (1575, 1625, 1030, 1050)
SLOT3_LABEL  = (1500, 1550, 1030, 1050)
SLOT_DETECT_FRAMES = 240

# Contour models live in two folders: models/clips/ (short, hand-clipped
# gameplay) and models/videos/ (long, full-length uploads such as random
# YouTube videos). The exact traced contour differs between the two, so the
# folder is chosen per source by duration — see contour_paths_for(). Resolved
# from this file's location so detection works regardless of the cwd.
MODELS_ROOT   = Path(__file__).resolve().parent.parent / "models"
CLIP_MAX_SECS = 60   # source <= 60s -> models/clips, longer -> models/videos

# Newer recordings ("training video N ...") use the RIGHT dash slot (slot2).
# Any video numbered >= this is forced to RIGHT regardless of duration or
# auto-detect; older numbers keep the auto-detect + long-video LEFT pin.
RIGHT_SLOT_MIN_VIDEO = 67


def video_number(name) -> int | None:
    """Extract N from a 'training video N ...' filename; None if it doesn't match."""
    m = re.search(r"training video\s+(\d+)", str(name), re.IGNORECASE)
    return int(m.group(1)) if m else None

WHITE_THRESH       = 200
WHITE_RATIO_THRESH = 0.95
LABEL_GREY_THRESH  = 110
ZOOM_LOW_THRESH    = 0.5
OFF_FRAMES         = 3
DASH_REARM_SECS    = 0.3

# combo window = 450(n-1) + 100 ms  →  Double:550ms, Triple:1000ms, Quad:1450ms, Penta:1900ms
COMBO_NAMES           = {2: "Double", 3: "Triple", 4: "Quad", 5: "Penta"}

# Intra-video parallelism: short clips are processed one-per-worker (the pool
# parallelises ACROSS clips). A long video is a single worker's job and would
# leave the other cores idle, so anything longer than CHUNK_MIN_SECS is split
# into N_CHUNKS contiguous frame ranges decoded in parallel. Only the per-frame
# measurement is split (every frame is measured independently, so a cut between
# frames changes nothing); the records are merged in frame order and the dash
# state machine runs ONCE over the reassembled sequence -> output is identical
# to processing the whole video in one pass.
CHUNK_MIN_SECS = 300     # only split videos longer than 5 min
N_CHUNKS       = 10      # contiguous pieces a long video is split into

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# ---------------------------------------------------------------------------


def open_video(video_path: Path):
    video_path = Path(video_path)          # callers may pass a str (e.g. RAM-staged path)
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap, None
    tmp = tempfile.NamedTemporaryFile(suffix=video_path.suffix, delete=False)
    tmp.close()
    shutil.copy2(str(video_path), tmp.name)
    return cv2.VideoCapture(tmp.name), tmp.name


def count_label_contours(frame, x0, x1, y0, y1) -> int:
    h, w = frame.shape[:2]
    crop = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    if crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, LABEL_GREY_THRESH, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(cnts)


def white_ratio_in_contours(frame, contours, x0, y0, x1, y1):
    h, w = frame.shape[:2]
    region = frame[min(y0,h):min(y1,h), min(x0,w):min(x1,w)]
    gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mask   = np.zeros(gray.shape, dtype=np.uint8)
    shifted = [(c - np.array([[[x0, y0]]])).astype(np.int32) for c in contours]
    cv2.drawContours(mask, shifted, -1, 255, thickness=cv2.FILLED)
    total_pixels = mask.sum() // 255
    if total_pixels == 0:
        return False, 0.0
    white_pixels = ((gray > WHITE_THRESH) & (mask > 0)).sum()
    ratio = white_pixels / total_pixels
    return ratio >= WHITE_RATIO_THRESH, ratio


def zoom_ratio_excluding_contours(frame, contours, sx0, sy0, sx1, sy1):
    h, w = frame.shape[:2]
    crop = frame[min(sy0,h):min(sy1,h), min(sx0,w):min(sx1,w)]
    if crop.size == 0:
        return 0.0
    gray_z    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    excl_mask = np.zeros(gray_z.shape, dtype=np.uint8)
    if contours:
        shifted = [(c - np.array([[[sx0, sy0]]])).astype(np.int32) for c in contours]
        cv2.drawContours(excl_mask, shifted, -1, 255, thickness=cv2.FILLED)
    outside = excl_mask == 0
    total   = outside.sum()
    return ((gray_z > WHITE_THRESH) & outside).sum() / total if total > 0 else 0.0


def load_contours(path_str):
    p = Path(path_str)
    if not p.exists():
        return []
    data = np.load(str(p), allow_pickle=True)
    return list(data)


def contour_paths_for(duration_secs: float):
    """Pick the contour folder by clip length: models/videos/ for sources
    longer than CLIP_MAX_SECS (full-length videos), models/clips/ otherwise.
    Returns (right, right_old, left, folder_name)."""
    cdir = MODELS_ROOT / ("videos" if duration_secs > CLIP_MAX_SECS else "clips")
    return (str(cdir / "slot_x_contour_right.npy"),
            str(cdir / "slot_x_contour_right_old.npy"),
            str(cdir / "slot_x_contour_left.npy"),
            cdir.name)


def _progress_bar(done, total, width=30, label="videos"):
    """One-line ASCII progress bar updated in place via carriage return."""
    total  = max(total, 1)
    frac   = done / total
    filled = int(width * frac)
    bar    = "#" * filled + "-" * (width - filled)
    end    = "\n" if done >= total else ""
    print(f"\r  [{bar}] {done}/{total} {label} ({frac * 100:5.1f}%)", end=end, flush=True)


def fmt_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _prepare(video_path: Path):
    """Open the video, pick the contour folder by duration, detect the dash
    slot, and build the candidate contours. Returns a dict the measurement and
    finalize steps need, or None if the video can't be opened. This is the
    cheap front-matter (reads only the first SLOT_DETECT_FRAMES) and is shared
    by the sequential and chunked paths."""
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        return None

    src_fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(src_fps / PROCESS_FPS))
    rearm_frames   = int(DASH_REARM_SECS * src_fps)

    # Choose the contour folder (clips vs videos) by source duration.
    src_n    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = src_n / src_fps if src_fps else 0.0
    right_path, right_old_path, left_path, cdir_name = contour_paths_for(duration)
    print(f"  [{video_path.name}] duration {duration:.1f}s -> contours from models/{cdir_name}/")

    # Detect which slot has the LSHIFT label by counting contours in the left
    # label box. Read the first SLOT_DETECT_FRAMES sequentially and sample every
    # 10th (seeking per-sample is ~2.6s/seek on this H.264 — far slower).
    left_cnts_acc = 0
    sampled = 0
    read = 0
    while read < SLOT_DETECT_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if read % 10 == 0:
            left_cnts_acc += count_label_contours(frame, *SLOT3_LABEL)
            sampled += 1
        read += 1
    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)

    avg_cnts = left_cnts_acc / max(sampled, 1)
    is_right = avg_cnts > 1

    # Newer recordings (training video >= RIGHT_SLOT_MIN_VIDEO) use the RIGHT slot
    # (slot2); force it regardless of duration / auto-detect.
    vnum        = video_number(video_path.name)
    force_right = vnum is not None and vnum >= RIGHT_SLOT_MIN_VIDEO
    forced_left = False
    if force_right:
        is_right = True
    else:
        # Long-form gameplay (> CLIP_MAX_SECS, i.e. the models/videos/ set) is
        # always LEFT-slot (slot3) in this older data. The auto-detector
        # occasionally misfires to RIGHT (slot2) on these, so detection would
        # fall back to a blank rectangle and return 0 dashes (e.g. videos 56,
        # 60). Pin long videos to LEFT so a misfire can't drop a whole video.
        # Short clips keep the auto-detect (both slots occur there).
        forced_left = duration > CLIP_MAX_SECS and is_right
        if forced_left:
            is_right = False

    slot_name = "RIGHT (slot2)" if is_right else "LEFT (slot3)"
    if force_right:
        note = f"  [forced RIGHT: video #{vnum} >= {RIGHT_SLOT_MIN_VIDEO}]"
    elif forced_left:
        note = "  [forced LEFT: long video]"
    else:
        note = ""
    print(f"  [{video_path.name}] Dash slot: {slot_name}  "
          f"(avg contours: {avg_cnts:.2f}){note}")

    # Candidate contours for the chosen slot. RIGHT has two possible UI
    # layouts (normal vs old); LEFT has one. We evaluate every candidate on
    # EVERY frame, record whether each is white (filled) + its zoom state, then
    # keep whichever contour is white on the most frames overall — the dash
    # icon only flashes briefly, so per-frame counting beats sparse sampling.
    if is_right:
        # Forced-RIGHT (>= RIGHT_SLOT_MIN_VIDEO) videos are the NORMAL layout, so
        # only the normal contour is a candidate. Auto-detected right (older short
        # clips) still tries normal + old and keeps whichever fills more.
        cand_specs = [("normal", load_contours(right_path))]
        if not force_right:
            cand_specs.append(("old", load_contours(right_old_path)))
        fallback = SLOT2_SEARCH
    else:
        cand_specs = [("left", load_contours(left_path))]
        fallback = SLOT3_SEARCH

    candidates = []
    for label, cnts in cand_specs:
        if cnts:
            rx, ry, rw, rh = cv2.boundingRect(np.concatenate(cnts).astype(np.int32))
            sx0, sx1, sy0, sy1 = rx, rx + rw, ry, ry + rh
        else:
            print(f"  [{video_path.name}] [warn] no '{label}' contour in "
                  f"models/{cdir_name}/ -> falling back to the search rectangle")
            sx0, sx1, sy0, sy1 = fallback
        candidates.append(dict(label=label, contours=cnts,
                               bbox=(sx0, sx1, sy0, sy1)))

    return dict(candidates=candidates, src_fps=src_fps,
                frame_interval=frame_interval, rearm_frames=rearm_frames,
                n_frames=src_n)


def _measure_range(video_path: Path, candidates, frame_interval,
                   start_frame=0, end_frame=10**12):
    """Decode frames [start_frame, end_frame) and record, for every candidate,
    whether it is white and whether zoom is low, at each frame_interval-th
    ABSOLUTE frame index. Returns (rec_idx, white_rec, zoom_rec) aligned by
    list position. Every measurement is single-frame, so splitting the video
    at any boundary cannot change a single value."""
    rec_idx   = []
    white_rec = [[] for _ in candidates]
    zoom_rec  = [[] for _ in candidates]

    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        return rec_idx, white_rec, zoom_rec

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    read_idx = start_frame
    while read_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if read_idx % frame_interval == 0:
            rec_idx.append(read_idx)
            for ci, st in enumerate(candidates):
                sx0, sx1, sy0, sy1 = st["bbox"]
                white_state, _ = white_ratio_in_contours(frame, st["contours"], sx0, sy0, sx1, sy1)
                zoom_ratio     = zoom_ratio_excluding_contours(frame, st["contours"], sx0, sy0, sx1, sy1)
                white_rec[ci].append(white_state)
                zoom_rec[ci].append(zoom_ratio < ZOOM_LOW_THRESH)
        read_idx += 1

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)
    return rec_idx, white_rec, zoom_rec


def _finalize(video_path: Path, candidates, rec_idx, white_rec, zoom_rec,
              src_fps, rearm_frames) -> tuple:
    """Pick the winning contour and run the dash state machine over its full
    recorded sequence. Identical whether the records came from one pass or from
    merged chunks."""
    # pick the contour that is white (filled) on the most frames
    fills = [sum(w) for w in white_rec]
    win = max(range(len(candidates)), key=lambda i: fills[i]) if candidates else 0
    if len(candidates) > 1:
        summary = ", ".join(f"{candidates[i]['label']}={fills[i]}" for i in range(len(candidates)))
        print(f"  [{video_path.name}] RIGHT layout: {candidates[win]['label']}  (white fills: {summary})")

    # --- verbatim dash state machine, run over the WINNER's recorded sequence ---
    off_streak   = 0
    was_off      = True
    rearm_at     = 0
    total_dashes = 0
    timestamps   = []
    dash_secs    = []

    combo_count      = 0
    combo_start_sec  = None
    combos           = []

    for read_idx, white_state, zoom_low in zip(rec_idx, white_rec[win], zoom_rec[win]):
        if white_state and zoom_low and read_idx >= rearm_at and was_off:
            total_dashes += 1
            t_sec = read_idx / src_fps
            timestamps.append(fmt_timestamp(t_sec))
            dash_secs.append(t_sec)
            rearm_at = read_idx + rearm_frames
            was_off  = False

            if combo_start_sec is None:
                combo_start_sec = t_sec
                combo_count     = 1
            else:
                new_count = combo_count + 1
                if (t_sec - combo_start_sec) <= 0.45 * (new_count - 1) + 0.275:
                    combo_count = new_count
                else:
                    if combo_count >= 2:
                        combos.append((combo_count, COMBO_NAMES.get(combo_count, f"{combo_count}x")))
                    combo_start_sec = t_sec
                    combo_count     = 1

        if not white_state:
            off_streak += 1
            if off_streak >= OFF_FRAMES:
                was_off = True
        else:
            off_streak = 0

    if combo_count >= 2:
        combos.append((combo_count, COMBO_NAMES.get(combo_count, f"{combo_count}x")))

    return video_path.name, total_dashes, timestamps, combos, dash_secs


def count_dashes(video_path: Path) -> tuple:
    """Sequential path: one worker decodes the whole video.
    Returns (video_name, total_dashes, timestamp_strings, combos, dash_secs)."""
    prep = _prepare(video_path)
    if prep is None:
        return video_path.name, 0, [], [], []
    rec_idx, white_rec, zoom_rec = _measure_range(
        video_path, prep["candidates"], prep["frame_interval"])
    return _finalize(video_path, prep["candidates"], rec_idx, white_rec, zoom_rec,
                     prep["src_fps"], prep["rearm_frames"])


def _chunk_worker(args):
    """Pool task: measure one contiguous frame range of a video. Returns the
    range's start frame (for ordering the merge) plus its records."""
    video_path_str, candidates, frame_interval, start_frame, end_frame = args
    recs = _measure_range(Path(video_path_str), candidates, frame_interval,
                          start_frame, end_frame)
    return start_frame, recs


def count_dashes_chunked(video_path: Path, pool, n_chunks: int = N_CHUNKS) -> tuple:
    """Parallel path for long videos: split the decode into n_chunks contiguous
    frame ranges across `pool`, merge the per-frame records in frame order, then
    run the single dash state machine. Output is identical to count_dashes()."""
    prep = _prepare(video_path)
    if prep is None:
        return video_path.name, 0, [], [], []

    candidates = prep["candidates"]
    fi         = prep["frame_interval"]
    n          = prep["n_frames"]

    if n <= 0 or n_chunks <= 1:
        # no reliable frame count -> fall back to a single sequential pass
        rec_idx, white_rec, zoom_rec = _measure_range(video_path, candidates, fi)
    else:
        # contiguous, non-overlapping ranges; the last reads to EOF so the tail
        # is covered even if the reported frame count is slightly off.
        bounds = [round(i * n / n_chunks) for i in range(n_chunks)] + [10**12]
        tasks  = [(str(video_path), candidates, fi, bounds[i], bounds[i + 1])
                  for i in range(n_chunks)]
        print(f"  [{video_path.name}] decoding {n} frames in {n_chunks} parallel chunks")
        parts = pool.map(_chunk_worker, tasks)
        parts.sort(key=lambda p: p[0])           # by start frame -> global order

        rec_idx   = []
        white_rec = [[] for _ in candidates]
        zoom_rec  = [[] for _ in candidates]
        for _start, (ri, wr, zr) in parts:
            rec_idx.extend(ri)
            for ci in range(len(candidates)):
                white_rec[ci].extend(wr[ci])
                zoom_rec[ci].extend(zr[ci])

    return _finalize(video_path, candidates, rec_idx, white_rec, zoom_rec,
                     prep["src_fps"], prep["rearm_frames"])


def _video_duration(video_path: Path) -> float:
    """Seconds, from frame count / fps. Cheap: opens the container but decodes
    nothing."""
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)
    return n / fps if fps else 0.0


def _is_long(video_path: Path) -> bool:
    return _video_duration(video_path) > CHUNK_MIN_SECS


def detect_all(videos, pool, on_done=None, on_result=None):
    """Run detection over `videos` using `pool`: short clips one-per-worker (the
    pool parallelises across them), then long videos one at a time with their
    decode split across the whole pool. Returns {video_name: result}. `on_done`
    is called with the running completed count after each video (for a progress
    bar); `on_result(name, result)` is called the moment each video finishes (so
    the caller can persist it before anything downstream can fail). Processing is
    clips-first, but results are keyed by name so the caller can restore order."""
    flags = [(v, _is_long(v)) for v in videos]   # one duration probe per video
    short = [v for v, lng in flags if not lng]
    long_ = [v for v, lng in flags if lng]
    print(f"  {len(short)} short (per-video parallel) + {len(long_)} long (chunked)")

    res_by_name = {}
    done = 0

    def _record(res):
        nonlocal done
        res_by_name[res[0]] = res
        done += 1
        if on_result:
            on_result(res[0], res)
        if on_done:
            on_done(done)

    # clips first: each clip is one pool task
    for res in pool.imap(_worker, [str(v) for v in short]):
        _record(res)
    # then long videos: each split into N_CHUNKS across the whole pool
    for v in long_:
        _record(count_dashes_chunked(v, pool))
    return res_by_name


def _worker(video_path_str: str) -> tuple:
    return count_dashes(Path(video_path_str))


def main():
    input_path = Path(INPUT_DIR)
    videos = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not videos:
        print(f"No videos found in '{INPUT_DIR}/'.")
        return

    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Processing {len(videos)} video(s) at {PROCESS_FPS} fps  |  workers: {num_workers}\n")
    print("=" * 60)

    with mp.Pool(processes=num_workers) as pool:
        res_by_name = detect_all(
            videos, pool,
            on_done=lambda done: _progress_bar(done, len(videos)))
    all_results = [res_by_name[v.name] for v in videos]

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Write timestamps text file
    txt_file = Path(OUTPUT_DIR) / "dash_timestamps.txt"
    results  = {}
    txt_lines = []

    for name, total, timestamps, combos, dash_secs in all_results:
        times_str  = ", ".join(timestamps) if timestamps else "none"
        combos_str = ", ".join(f"{lbl} ({n})" for n, lbl in combos) if combos else "none"
        print(f"[{name}]")
        print(f"  Dashes : {total}")
        print(f"  Times  : {times_str}")
        print(f"  Combos : {combos_str}\n")
        results[name] = {"dashes": total, "timestamps": timestamps, "combos": [[n, lbl] for n, lbl in combos]}

        txt_lines.append(f"=== {name} ===")
        if dash_secs:
            for i, (t, ts) in enumerate(zip(dash_secs, timestamps)):
                if i == 0:
                    txt_lines.append(f"  dash {i+1:>2}: {ts}")
                else:
                    delta = t - dash_secs[i - 1]
                    txt_lines.append(f"  dash {i+1:>2}: {ts}  (delta: {delta:.3f}s)")
        else:
            txt_lines.append("  no dashes detected")
        txt_lines.append("")

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    out_file = Path(OUTPUT_DIR) / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"Timestamps : {txt_file}")
    print(f"JSON       : {out_file}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
