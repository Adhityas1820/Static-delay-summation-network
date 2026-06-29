"""
make_dash_videos.py  —  build the `dash-videos/` upload folder (Stage 1, local)
===============================================================================
Creates ./dash-videos/ containing:
  * a copy of the reviewed dash_intervals.csv (the Stage-1 labels), and
  * every video in ./videos/ re-encoded to a small H.264 copy.

Stage 2 resizes every frame to IMG_SIZE x IMG_SIZE (224) anyway, so the full
1080p clips are wasteful to upload to Kaggle (the real wall-clock bottleneck).
We downscale to 360p — comfortably above 224 — which shrinks the upload ~15x with
no effect on the features.

CRITICAL invariants (or the labels misalign / videos get dropped):
  * SAME filename INCLUDING extension (.mkv stays .mkv) — Stage 2 matches videos
    to CSV rows by exact name.
  * SAME frame count + fps — labels are per-frame. We use -fps_mode passthrough
    (1 frame in -> 1 frame out), set no -r, and VERIFY the frame count after each
    encode; a mismatch keeps the temp file for inspection and skips that video.

INCREMENTAL: a video already present in dash-videos/ with a matching frame count
is skipped, so re-running Stage 1 after a CSV edit just re-copies the (instant)
CSV and re-compresses only new/changed clips. Use force=True / --force to redo all.

Normally driven by stage1.py (which calls build() after transcode + labels), so a
single `python stage1.py` leaves dash-videos/ ready to upload. Direct use:
    python -m dash_code.make_dash_videos                 # 360p, CRF 28
    python -m dash_code.make_dash_videos --height 240 --crf 28 --force
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

# Allow `python -m dash_code.make_dash_videos` (package context) and direct runs.
if __package__ in (None, ""):
    import sys
    _here = os.path.abspath(__file__)
    sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))

from . import config as C

SRC_DIR = Path(C.RAW_VIDEO_DIR)
OUT_DIR = C.PROJECT_ROOT / "dash-videos"
CSV_SRC = Path(C.LABELS_CSV)

FFMPEG  = shutil.which("ffmpeg")  or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".webm")


def frame_count(path: Path):
    """Reliable frame count via packet count (fast, no full decode). None if
    unavailable."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


def _cmd(src: Path, dst: Path, height: int, crf: int, hwaccel: bool):
    cmd = [FFMPEG, "-y"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]            # GPU-accelerated DECODE (scale stays on CPU)
    cmd += ["-i", str(src), "-map", "0:v:0",
            "-vf", f"scale=-2:{height}",       # downscale, keep aspect, even width
            "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-fps_mode", "passthrough",        # 1 frame in -> 1 frame out
            "-an", str(dst)]
    return cmd


def encode(src: Path, dst: Path, height: int, crf: int, hwaccel: bool) -> bool:
    """GPU-accelerated decode + libx264 encode; falls back to pure CPU on error."""
    r = None
    for hw in ([True, False] if hwaccel else [False]):
        r = subprocess.run(_cmd(src, dst, height, crf, hw),
                           capture_output=True, text=True)
        if r.returncode == 0:
            return True
        print(f"     attempt (hwaccel={hw}) failed; falling back...", flush=True)
    if r is not None:
        print(f"     [!] ffmpeg failed:\n{r.stderr.strip()[-400:]}")
    return False


def build(height: int = 360, crf: int = 28, hwaccel: bool = True,
          force: bool = False) -> tuple:
    """Compress every ./videos clip into ./dash-videos/ (frame-preserving, same
    filenames) and copy the reviewed CSV alongside. Incremental unless force.
    Returns (ok, skipped, bad) filename lists. Importable so stage1.py drives it."""
    if not SRC_DIR.is_dir():
        print(f"No {SRC_DIR}/ — set DASH_RAW_DIR or add videos.")
        return [], [], []
    vids = sorted(p for p in SRC_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not vids:
        print(f"No videos in {SRC_DIR}.")
        return [], [], []
    OUT_DIR.mkdir(exist_ok=True)

    # copy the reviewed CSV (the Stage-1 labels) alongside the videos
    if CSV_SRC.exists():
        shutil.copy2(CSV_SRC, OUT_DIR / CSV_SRC.name)
        print(f"CSV -> {OUT_DIR / CSV_SRC.name}")
    else:
        print(f"[!] {CSV_SRC} not found — run the labels stage first; "
              f"videos will still be made.")

    print(f"Compressing {len(vids)} video(s) to {height}p CRF {crf} -> {OUT_DIR}/  "
          f"(frame-preserving, same filenames{'' if not force else ', FORCE'})\n")
    ok, skipped, bad = [], [], []
    for i, p in enumerate(vids, 1):
        dst = OUT_DIR / p.name                 # SAME name + extension
        src_n = frame_count(p)
        if not force and dst.exists() and src_n is not None and frame_count(dst) == src_n:
            skipped.append(p.name)
            print(f"  [{i}/{len(vids)}] {p.name}  already current ({src_n} frames) — skip")
            continue
        tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
        print(f"  [{i}/{len(vids)}] {p.name}  ({src_n} frames) ...", flush=True)
        if not encode(p, tmp, height, crf, hwaccel):
            tmp.unlink(missing_ok=True); bad.append(p.name); continue
        dst_n = frame_count(tmp)
        if src_n is None or dst_n != src_n:
            print(f"     -> {dst_n} frames: FRAME COUNT CHANGED/UNVERIFIABLE — "
                  f"keeping temp {tmp.name} for inspection, skipping.")
            bad.append(p.name); continue
        os.replace(tmp, dst)
        mb_in, mb_out = p.stat().st_size / 1e6, dst.stat().st_size / 1e6
        print(f"     -> {dst.name}  ({dst_n} frames)  {mb_in:.0f}MB -> {mb_out:.0f}MB")
        ok.append(p.name)

    print(f"\nVideos: {len(ok)} compressed, {len(skipped)} already current, "
          f"{len(bad)} failed/unverified.")
    if bad:
        print(f"  [!] {bad}")
    return ok, skipped, bad


def main():
    ap = argparse.ArgumentParser(description="Build the dash-videos/ upload folder.")
    ap.add_argument("--height", type=int, default=360, help="target height (default 360)")
    ap.add_argument("--crf", type=int, default=28, help="libx264 CRF (default 28)")
    ap.add_argument("--no-hwaccel", action="store_true", help="disable GPU decode")
    ap.add_argument("--force", action="store_true", help="re-encode even if up to date")
    args = ap.parse_args()
    build(args.height, args.crf, not args.no_hwaccel, args.force)
    print(f"\nUpload {OUT_DIR}/ as the `dash-videos` Kaggle dataset "
          f"(videos + dash_intervals.csv).")


if __name__ == "__main__":
    main()
