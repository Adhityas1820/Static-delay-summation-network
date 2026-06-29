"""
process_data.py  —  STAGE 1 (local) + STAGE 2 (Kaggle) of the SDSNN dash detector
=================================================================================
Turns raw Marvel Rivals gameplay videos into per-frame ResNet18 features (the
spatial front-end) and per-frame dash/no-dash labels for the SDSNN.

It runs in two stages so the labels can be eyeballed BEFORE they become
training data (`--stage labels` = Stage 1, local; `--stage features` = Stage 2,
best run on Kaggle GPU):

  STAGE 1 (labels)   run the reused dash detector (helper/dash_counter.py, the
                     exact Marvel Rivals classifier file, with its parallel
                     multiprocessing pool) -> dash START points -> intervals
                     (start-50ms .. start+430ms) -> write processed/
                     dash_intervals.csv, then render annotated overlay videos
                     (also in parallel).  >>> STOP and review the CSV here. <<<

  STAGE 2 (features) read the (possibly hand-corrected) CSV, extract per-frame
                     ResNet18 features (512-d), build per-frame labels from the
                     CSV intervals, and save processed/features/<video>.npz.

Overlay videos (review/overlay/) box the dash window red, the exact detected
start frame yellow ("START"), and show each frame's index + timestamp so you
can cross-check the CSV.

Usage (or just run stage1.py at the repo root, which calls the labels stage):
    python -m dash_code.process_data                 # stage 1 (default): CSV + overlays, then stop
    python -m dash_code.process_data --no-overlay    # stage 1 without overlay videos (faster)
    python -m dash_code.process_data --clips         # stage 1 + per-interval clips too
    python -m dash_code.process_data --stage features            # after reviewing CSV
    python -m dash_code.process_data --stage all                 # both, no stop

Paths come from dash_code/config.py (overridable via env vars). Put videos
in ./videos (or set DASH_RAW_DIR). Detector contours live in ./models.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

# Allow running this file directly (`python dash_code/process_data.py`) as
# well as as a module (`python -m dash_code.process_data`). When run
# directly there is no package context, so put the repo root on sys.path.
if __package__ in (None, ""):
    import sys
    _here = os.path.abspath(__file__)
    sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))

import cv2
import numpy as np

from . import config as C
from . import dash_counter as dc, features as feat, labels as lab, io_utils


# --- detection cache -------------------------------------------------------
# Detection is the expensive part (a long video can take many minutes). Each
# finished video is written here immediately, so a crash in a LATER step (CSV,
# overlays) never throws away detection work — a re-run reuses anything still
# valid. An entry is valid only while the source file is unchanged (size+mtime)
# AND no contour model is newer than it (re-tracing a contour re-detects).
DETECT_CACHE = "detections.json"


def _video_sig(v: Path) -> list:
    st = v.stat()
    return [int(st.st_size), int(st.st_mtime)]


def _models_mtime() -> float:
    root = Path(C.MODELS_DIR)
    times = [p.stat().st_mtime for p in root.rglob("*.npy")] if root.exists() else []
    return max(times) if times else 0.0


def _load_detect_cache() -> dict:
    path = Path(C.PROCESSED_DIR) / DETECT_CACHE
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_detect_cache(cache: dict):
    path = Path(C.PROCESSED_DIR) / DETECT_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    
    # Retry atomic swap to handle transient Windows lock errors
    for _ in range(10):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            time.sleep(0.1)
    else:
        os.replace(tmp, path)   # If it still fails after retries, let it crash normally



def list_videos() -> list:
    d = Path(C.RAW_VIDEO_DIR)
    if not d.exists():
        return []
    return sorted(f for f in d.iterdir()
                  if f.is_file() and f.suffix.lower() in C.VIDEO_EXTENSIONS)


def _probe_fps_frames(video_path: Path):
    cap, tmp = dc.open_video(video_path)
    if not cap.isOpened():
        return float(C.FPS), 0
    fps = cap.get(cv2.CAP_PROP_FPS) or C.FPS
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if tmp:
        Path(tmp).unlink(missing_ok=True)
    return float(fps), max(n, 0)


# ---------------------------------------------------------------------------
# STAGE 1 — detect dashes (parallel), build intervals, write the review CSV,
#           then render overlays (parallel)
# ---------------------------------------------------------------------------
def stage_labels(videos: list, export_clips: bool, export_overlay: bool,
                 redetect: bool = False):
    n_workers = max(1, mp.cpu_count() - 1)

    # --- reuse cached detections for unchanged videos ----------------------
    cache    = {} if redetect else _load_detect_cache()
    models_t = _models_mtime()
    fresh, todo = {}, []
    for v in videos:
        ent = cache.get(v.name)
        if ent and ent.get("sig") == _video_sig(v) and ent.get("detected_at", 0) >= models_t:
            fresh[v.name] = ent["result"]
        else:
            todo.append(v)
    if fresh:
        print(f"Reusing cached detection for {len(fresh)}/{len(videos)} unchanged "
              f"video(s).  ({len(todo)} to (re)detect)")

    # --- parallel detection of the remaining videos ------------------------
    # Short clips run one-per-worker; long videos (>5 min) get their decode
    # split across the whole pool. Each finished video is cached immediately so
    # a later crash can't lose it. Results are keyed by name -> restore order.
    if todo:
        todo_by_name = {v.name: v for v in todo}

        def _on_result(name, result):
            cache[name] = dict(sig=_video_sig(todo_by_name[name]),
                               detected_at=time.time(), result=result)
            _save_detect_cache(cache)

        print(f"Detecting dashes in parallel  |  workers: {n_workers}")
        with mp.Pool(processes=n_workers) as pool:
            new_results = dc.detect_all(
                todo, pool,
                on_done=lambda done: dc._progress_bar(done, len(todo)),
                on_result=_on_result)
        fresh.update(new_results)

    results = [fresh[v.name] for v in videos]

    # --- build per-video detections + CSV rows ---
    detections = {}
    all_rows = []
    for v, res in zip(videos, results):
        _name, _total, _ts, _combos, dash_secs = res
        fps, probe_n = _probe_fps_frames(v)
        dash_frames = [int(round(s * fps)) for s in dash_secs]
        last = (max(dash_frames) + lab.POST_FRAMES + 1) if dash_frames else 1
        n_frames = max(probe_n, last)
        det = dict(video=v.name, src_fps=fps, n_frames=n_frames,
                   dash_frames=dash_frames, dash_secs=dash_secs)
        detections[v.name] = det
        all_rows.extend(lab.interval_csv_rows(det))
        print(f"  [{v.name}] {len(dash_frames)} dash(es), {n_frames} frames")

    try:
        csv_path = io_utils.write_intervals_csv(all_rows)
    except PermissionError:
        print(f"\n[!] Can't write {C.LABELS_CSV} — it's open in another program "
              f"(Excel / Notepad / OneDrive sync). Close it and re-run.\n"
              f"    Detection is cached, so the re-run will be quick.")
        return
    print(f"\nWrote {len(all_rows)} interval rows -> {csv_path}")

    # --- overlays / clips, after detection, in parallel ---
    if export_overlay:
        print(f"\nRendering overlay videos in parallel  |  workers: {n_workers}")
        with mp.Pool(processes=n_workers) as pool:
            pool.starmap(_export_overlay, [(v, detections[v.name]) for v in videos])
    if export_clips:
        print(f"\nExporting per-interval clips in parallel  |  workers: {n_workers}")
        with mp.Pool(processes=n_workers) as pool:
            pool.starmap(_export_interval_clips, [(v, detections[v.name]) for v in videos])

    print("\n>>> Review the CSV (and any clips/overlay) before running "
          "`--stage features`. Edit/delete wrong rows; they ARE the labels. <<<")


def _interval_frames(det: dict) -> list:
    return [lab.build_interval(f, det["n_frames"]) for f in det["dash_frames"]]


def _export_interval_clips(video_path: Path, det: dict):
    out_dir = Path(C.REVIEW_DIR) / "clips" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cap, tmp = dc.open_video(video_path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or C.FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for i, (s, e) in enumerate(_interval_frames(det)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        writer = cv2.VideoWriter(str(out_dir / f"dash{i+1:02d}_{s}-{e}.mp4"),
                                 fourcc, fps, (w, h))
        for _ in range(e - s + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()

    cap.release()
    if tmp:
        Path(tmp).unlink(missing_ok=True)
    print(f"  clips -> {out_dir}")


def _export_overlay(video_path: Path, det: dict):
    out_dir = Path(C.REVIEW_DIR) / "overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = det["n_frames"]

    # per-frame lookups: which dash # an interval frame belongs to, and the
    # exact detected start frames (marked distinctly from the padded window).
    dash_of   = [0] * n
    start_set = set(f for f in det["dash_frames"] if 0 <= f < n)
    comp_set  = set(lab.completion_frame(f, n) for f in det["dash_frames"])
    for i, (s, e) in enumerate(_interval_frames(det), start=1):
        for f in range(max(0, s), min(n - 1, e) + 1):
            dash_of[f] = i

    cap, tmp = dc.open_video(video_path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or C.FPS
    w, h = C.REVIEW_W, C.REVIEW_H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = out_dir / f"{video_path.stem}_overlay.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (w, h))

        in_dash = idx < n and dash_of[idx] > 0
        is_comp = idx in comp_set
        if in_dash or is_comp:
            is_start = idx in start_set
            # green completion (the spike center / training peak) takes priority,
            # then yellow start, then red dash window.
            if is_comp:
                colour, thick, lab_tag = (0, 255, 0), 14, "  COMPLETE"
            elif is_start:
                colour, thick, lab_tag = (0, 255, 255), 14, "  START"
            else:
                colour, thick, lab_tag = (0, 0, 255), 6, ""
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, thick)
            dash_n = dash_of[idx] if in_dash else 0
            cv2.putText(frame, f"DASH #{dash_n}{lab_tag}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

        # always-on readout for cross-checking the CSV
        ts = lab.fmt_ts(idx * C.MS_PER_FRAME)
        cv2.putText(frame, f"f{idx}  {ts}", (30, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)
        idx += 1

    writer.release()
    cap.release()
    if tmp:
        Path(tmp).unlink(missing_ok=True)
    print(f"  overlay -> {out_path}")


# ---------------------------------------------------------------------------
# STAGE 2 — extract features, build labels from the reviewed CSV, save arrays
# ---------------------------------------------------------------------------
def _reuse_features(v, force: bool):
    """If a usable .npz already exists for v, return (features, src_fps, old_pos)
    so the expensive decode can be skipped; else None. Features are reused only
    when the cached frame count still matches the current video exactly, so
    labels rebuilt from the CSV stay aligned. `--force` ignores any cache."""
    npz_path = C.FEATURES_DIR / f"{v.stem}.npz"
    if force or not npz_path.exists():
        return None
    try:
        z = np.load(npz_path)
        feats   = z["features"]
        src_fps = float(z["src_fps"])
        old_pos = int(round(float(z["labels"].sum())))   # soft mass; see stage_features
    except Exception:
        return None
    _, cur_frames = _probe_fps_frames(v)
    if feats.shape[0] != cur_frames:
        return None                      # video changed -> must re-extract
    return feats, src_fps, old_pos


def stage_features(videos: list, force: bool = False, shard: tuple = None):
    by_video = io_utils.read_intervals_csv()
    centers_by_video = io_utils.read_completion_centers()
    if not by_video:
        print(f"No intervals found in {C.LABELS_CSV}. Run `--stage labels` first.")
        return

    # multi-GPU: each worker handles videos[i::N] (round-robin balances long clips)
    if shard is not None:
        i, N = shard
        videos = videos[i::N]
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        print(f"[shard {i}/{N}] CUDA_VISIBLE_DEVICES={gpu}: {len(videos)} video(s)")

    manifest = []
    n_extract = n_relabel = n_uptodate = 0
    for v in videos:
        if v.name not in by_video:
            print(f"  [{v.name}] not in CSV — skipping "
                  f"(run labels stage or add it manually).")
            continue
        centers = centers_by_video.get(v.name, [])

        # Reuse cached features (the expensive part) when possible; only the new
        # videos actually get decoded. Labels are always rebuilt from the current
        # CSV, so a stale-labelled .npz is refreshed without a re-decode.
        reuse = _reuse_features(v, force)
        if reuse is None:
            print(f"  [{v.name}] extracting features...", flush=True)
            features = feat.extract_cnn_features(v)
            n_frames = features.shape[0]
            if n_frames == 0:
                print(f"  [!!] {v.name}: 0 frames decoded — this machine can't "
                      f"decode this video's codec (e.g. AV1). SKIPPED, no .npz "
                      f"written. Transcode it to H.264 and re-run, or it is "
                      f"silently lost from training.")
                continue
            src_fps, _ = _probe_fps_frames(v)
            old_pos = None
        else:
            features, src_fps, old_pos = reuse
            n_frames = features.shape[0]

        # soft Gaussian completion target (the temporal reframe), built from the
        # reviewed completion centers. n_pos is the soft mass (used only to detect
        # whether labels changed); n_peaks is the meaningful number — #dashes.
        y = lab.gaussian_labels(centers, n_frames)
        n_pos   = int(round(float(y.sum())))
        n_peaks = sum(1 for c in centers if 0 <= c < n_frames)

        if reuse is not None and old_pos == n_pos:
            tag = "up to date"                       # features + labels unchanged
            n_uptodate += 1
        else:
            out = io_utils.save_processed(v.stem, features, y, src_fps,
                                          feat.FEATURE_NAMES)
            if reuse is None:
                tag = f"extracted -> {out.name}"
                n_extract += 1
            else:
                tag = f"relabeled (features reused)"
                n_relabel += 1

        manifest.append(dict(
            video=v.name, n_frames=n_frames, n_peaks=n_peaks, n_pos=n_pos,
            pos_frac=round(n_pos / max(n_frames, 1), 4),
            n_features=features.shape[1],
        ))
        print(f"  [{v.name}] {tag}: frames={n_frames} dashes={n_peaks} "
              f"(soft-mass {n_pos})")

    print(f"\nFeature pass: {n_extract} extracted (decoded), "
          f"{n_relabel} relabeled (no decode), {n_uptodate} already current.")

    if manifest:
        if shard is not None:
            # a worker: write a partial manifest the launcher will merge
            mpath = C.PROCESSED_DIR / f"manifest_shard{shard[0]}.csv"
            io_utils.write_manifest(manifest, mpath)
            print(f"[shard {shard[0]}/{shard[1]}] {len(manifest)} rows -> {mpath.name}")
        else:
            mpath = io_utils.write_manifest(manifest)
            tot   = sum(m["n_frames"] for m in manifest)
            peaks = sum(m["n_peaks"] for m in manifest)
            print(f"\nManifest -> {mpath}")
            print(f"Totals: {tot} frames, {peaks} dash completions (peaks). "
                  f"Stage 3 trains heatmap regression on the soft target, not "
                  f"per-frame BCE — no pos-weight needed.")


# ---------------------------------------------------------------------------
# multi-GPU Stage 2: split the video list across GPUs (one worker per GPU)
# ---------------------------------------------------------------------------
def _auto_gpus() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def _parse_shard(s):
    if not s:
        return None
    i, n = s.split("/")
    return int(i), int(n)


def extract_features_multi(videos: list, n_gpus: int, force: bool):
    """Run Stage 2 across n_gpus GPUs by launching one worker process per GPU,
    each pinned to its card via CUDA_VISIBLE_DEVICES and handling videos[i::N].
    Subprocesses (not fork) keep each GPU's CUDA context fully isolated. The
    workers write partial manifests which we merge here."""
    import csv
    import subprocess
    import sys

    print(f"=== STAGE 2: ResNet18 features across {n_gpus} GPUs "
          f"(video list split round-robin) ===")
    base = [sys.executable, "-m", "dash_code.process_data", "--stage", "features"]
    procs = []
    for i in range(n_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)        # this worker sees only GPU i
        cmd = base + ["--shard", f"{i}/{n_gpus}"]
        if force:
            cmd.append("--force-features")
        print(f"  GPU {i} -> {' '.join(cmd[2:])}")
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    if any(r != 0 for r in rcs):
        print(f"[!] a GPU worker exited non-zero: {rcs} — check the logs above.")

    # merge the partial manifests the workers wrote
    rows = []
    for i in range(n_gpus):
        sp = C.PROCESSED_DIR / f"manifest_shard{i}.csv"
        if sp.exists():
            with open(sp, newline="", encoding="utf-8") as f:
                rows.extend(list(csv.DictReader(f)))
            sp.unlink()
    if rows:
        io_utils.write_manifest(rows)
        tot   = sum(int(r["n_frames"]) for r in rows)
        peaks = sum(int(r["n_peaks"]) for r in rows)
        print(f"\nManifest -> {C.PROCESSED_DIR / 'manifest.csv'}  "
              f"({len(rows)} videos, {tot} frames, {peaks} dash completions)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage 1/2: build SDSNN dash data.")
    ap.add_argument("--stage", choices=["labels", "features", "all"],
                    default="labels", help="default: labels (stops for review)")
    ap.add_argument("--clips", action="store_true",
                    help="also export per-interval review clips (labels stage)")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip the annotated overlay videos (labels stage)")
    ap.add_argument("--redetect", action="store_true",
                    help="ignore the detection cache and re-detect every video")
    ap.add_argument("--force-features", action="store_true",
                    help="re-extract every video's features (ignore cached .npz); "
                         "default reuses existing features and only decodes new videos")
    ap.add_argument("--gpus", type=int, default=0,
                    help="GPUs for Stage 2 feature extraction (0 = auto-detect). "
                         ">1 splits the video list across that many GPUs.")
    ap.add_argument("--shard", default=None,
                    help="internal: process only videos[i::N] (set by the "
                         "multi-GPU launcher; e.g. '0/2')")
    args = ap.parse_args()
    export_overlay = not args.no_overlay

    videos = list_videos()
    if not videos:
        print(f"No videos in {C.RAW_VIDEO_DIR}. Add clips (1920x1080 @ 60 fps) "
              f"or set DASH_RAW_DIR.")
        return
    print(f"Found {len(videos)} video(s) in {C.RAW_VIDEO_DIR}\n")

    if args.stage in ("labels", "all"):
        print("=== STAGE 1: detect dashes -> intervals -> review CSV ===")
        stage_labels(videos, args.clips, export_overlay, redetect=args.redetect)
    if args.stage in ("features", "all"):
        shard = _parse_shard(args.shard)
        n_gpus = args.gpus if args.gpus else _auto_gpus()
        if shard is None and n_gpus > 1:
            extract_features_multi(videos, n_gpus, force=args.force_features)
        else:
            print("\n=== STAGE 2: ResNet18 frame features + per-frame labels ===")
            stage_features(videos, force=args.force_features, shard=shard)


if __name__ == "__main__":
    mp.freeze_support()
    main()
