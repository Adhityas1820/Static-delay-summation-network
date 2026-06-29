"""
make_dash_videos.py  —  build the `dash-videos/` upload folder
==============================================================
One-time dataset-prep tool (NOT part of the pipeline). Creates ./dash-videos/
containing:
  * a copy of the reviewed dash_intervals.csv (the Stage-1 labels), and
  * every video in ./videos/ re-encoded to a small H.264 copy.

Stage 2 resizes every frame to IMG_SIZE x IMG_SIZE (224) anyway, so the full
1080p clips are wasteful to upload to Kaggle (the real wall-clock bottleneck).
We downscale to 360p — comfortably above 224 — which shrinks the upload ~15-20x
with no effect on the features.

CRITICAL invariants (or the labels misalign / videos get dropped):
  * SAME filename INCLUDING extension (.mkv stays .mkv) — Stage 2 matches videos
    to CSV rows by exact name.
  * SAME frame count + fps — labels are per-frame. We use -fps_mode passthrough
    (1 frame in -> 1 frame out), set no -r, and VERIFY the frame count after each
    encode; a mismatch keeps the temp file for inspection and skips that video.

Usage:
    python make_dash_videos.py                 # 360p, CRF 28 (default)
    python make_dash_videos.py --height 240 --crf 28
    python make_dash_videos.py --no-hwaccel    # disable GPU-accelerated decode
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from dash_code import config as C

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
    for hw in ([True, False] if hwaccel else [False]):
        r = subprocess.run(_cmd(src, dst, height, crf, hw),
                           capture_output=True, text=True)
        if r.returncode == 0:
            return True
        print(f"     attempt (hwaccel={hw}) failed; falling back...", flush=True)
    if r is not None:
        print(f"     [!] ffmpeg failed:\n{r.stderr.strip()[-400:]}")
    return False


def main():
    ap = argparse.ArgumentParser(description="Build the dash-videos/ upload folder.")
    ap.add_argument("--height", type=int, default=360, help="target height (default 360)")
    ap.add_argument("--crf", type=int, default=28, help="libx264 CRF (default 28)")
    ap.add_argument("--no-hwaccel", action="store_true", help="disable GPU decode")
    args = ap.parse_args()

    if not SRC_DIR.is_dir():
        print(f"No {SRC_DIR}/ — set DASH_RAW_DIR or add videos.")
        return
    vids = sorted(p for p in SRC_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not vids:
        print(f"No videos in {SRC_DIR}.")
        return
    OUT_DIR.mkdir(exist_ok=True)

    # copy the reviewed CSV (the Stage-1 labels) alongside the videos
    if CSV_SRC.exists():
        shutil.copy2(CSV_SRC, OUT_DIR / CSV_SRC.name)
        print(f"CSV -> {OUT_DIR / CSV_SRC.name}")
    else:
        print(f"[!] {CSV_SRC} not found — run Stage 1 first; videos will still be made.")

    print(f"Compressing {len(vids)} video(s) to {args.height}p CRF {args.crf} "
          f"-> {OUT_DIR}/  (frame-preserving, same filenames)\n")
    ok, bad = [], []
    for i, p in enumerate(vids, 1):
        src_n = frame_count(p)
        dst = OUT_DIR / p.name                 # SAME name + extension
        tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
        print(f"  [{i}/{len(vids)}] {p.name}  ({src_n} frames) ...", flush=True)
        if not encode(p, tmp, args.height, args.crf, not args.no_hwaccel):
            tmp.unlink(missing_ok=True); bad.append(p.name); continue
        dst_n = frame_count(tmp)
        if src_n is None or dst_n != src_n:
            print(f"     -> {dst_n} frames: FRAME COUNT CHANGED/UNVERIFIABLE — "
                  f"keeping temp {tmp.name} for inspection, skipping.")
            bad.append(p.name); continue
        import os
        os.replace(tmp, dst)
        mb_in, mb_out = p.stat().st_size / 1e6, dst.stat().st_size / 1e6
        print(f"     -> {dst.name}  ({dst_n} frames)  {mb_in:.0f}MB -> {mb_out:.0f}MB")
        ok.append(p.name)

    print(f"\nDone. {len(ok)} compressed, {len(bad)} failed/unverified.")
    if bad:
        print(f"  [!] {bad}")
    print(f"\nUpload {OUT_DIR}/ as the `dash-videos` Kaggle dataset "
          f"(videos + dash_intervals.csv).")


if __name__ == "__main__":
    main()
