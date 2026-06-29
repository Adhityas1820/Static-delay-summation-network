"""
transcode_av1.py  —  make AV1 videos decodable (Stage 1 locally + Stage 2 Kaggle)
=================================================================================
OpenCV/decord (and Kaggle's builds) can't decode AV1, so AV1 clips silently
extract 0 frames and vanish from training — and the local dash detector can't
read them either. This re-encodes every AV1 video in ./videos to H.264 at the
SAME fps, so the frame count is preserved 1:1 and the existing (hand-reviewed)
dash_intervals.csv labels stay aligned — no re-labeling needed.

Two output modes:
  (default)    H.264 copies -> ./videos_h264 with the SAME filename; originals
               in ./videos are NEVER touched. Slot straight into Kaggle.
  --inplace    Replace each AV1 file in ./videos with its H.264 version (same
               name). The original is removed ONLY after the new file is written
               AND its frame count is verified to match — a failed transcode
               leaves the original untouched.

Speed (decode AND encode default to the GPU):
  (default)    Full GPU: av1_cuvid decodes and h264_nvenc encodes — the CPU is
               barely touched. Fastest path. h264_nvenc runs at CQ 19 (visually
               lossless for the dash VFX, which Stage 2 downscales to 224 anyway).
  --hwdec      GPU decode (av1_cuvid) + CPU libx264 encode at CRF 16 (matches the
               quality of videos already transcoded by this tool).
  --cpu        Pure CPU libx264 (no GPU at all).
  Each mode falls back toward the CPU per-file if a GPU stage errors, so a
  transient NVDEC hiccup never stalls the batch.

After each file it re-probes the frame count and FAILS LOUDLY if it changed
(it shouldn't, since the sources are constant-frame-rate).

Usage:   python -m dash_code.transcode_av1            # full GPU -> videos_h264/
         python -m dash_code.transcode_av1 --inplace  # full GPU, overwrite videos/

Normally driven by stage1.py at the repo root (which calls transcode(inplace=True)
before the local labels stage); these direct invocations are for one-offs.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

# Allow running directly (`python dash_code/transcode_av1.py`) or as a
# module (`python -m dash_code.transcode_av1`). Without a package context,
# put the repo root on sys.path so `from helper... import config` works.
if __package__ in (None, ""):
    import sys
    _here = os.path.abspath(__file__)
    sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))

from . import config as C

# Paths come from config (PROJECT_ROOT-anchored), so this works from any cwd —
# e.g. when stage1.py drives it. Originals live in videos/; the non-inplace
# copies go to videos_h264/ next to them.
SRC_DIR = Path(C.RAW_VIDEO_DIR)
OUT_DIR = C.PROJECT_ROOT / "videos_h264"
CRF     = "16"          # libx264 near-lossless; dash VFX preserved
PRESET  = "veryfast"    # libx264 preset — these are long videos, keep it quick
CQ      = "19"          # h264_nvenc constant-quality (visually lossless here)
NVPRESET = "p5"         # h264_nvenc preset p1(fast)..p7(best); p5 = balanced

FFMPEG  = shutil.which("ffmpeg")  or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def probe(path, *entries):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=" + ",".join(entries),
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True).stdout.split()
    return out


def _ff_cmd(src: Path, dst: Path, decode_gpu: bool, encode_gpu: bool):
    cmd = [FFMPEG, "-y"]
    if decode_gpu:
        cmd += ["-c:v", "av1_cuvid"]               # NVDEC AV1 decode
    cmd += ["-i", str(src), "-map", "0:v:0"]
    if encode_gpu:
        cmd += ["-c:v", "h264_nvenc", "-preset", NVPRESET,
                "-rc", "vbr", "-cq", CQ, "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-crf", CRF, "-preset", PRESET]
    cmd += ["-pix_fmt", "yuv420p",
            "-fps_mode", "passthrough",            # 1 frame in -> 1 frame out
            "-an", str(dst)]
    return cmd


def _encode(src: Path, dst: Path, mode: str):
    """Re-encode src -> dst (H.264, frame-preserving). Returns True on success.
    mode 'gpu' = av1_cuvid + h264_nvenc, 'hwdec' = av1_cuvid + libx264,
    'cpu' = libx264. Each falls back toward the CPU per file if a GPU stage
    errors (e.g. a transient NVDEC session hiccup), so the batch never stalls."""
    ladder = {
        "gpu":   [(True, True), (False, True), (False, False)],
        "hwdec": [(True, False), (False, False)],
        "cpu":   [(False, False)],
    }[mode]
    r = None
    for dg, eg in ladder:
        r = subprocess.run(_ff_cmd(src, dst, dg, eg),
                           capture_output=True, text=True)
        if r.returncode == 0:
            return True
        print(f"     attempt (gpu_decode={dg}, gpu_encode={eg}) failed; "
              f"falling back...", flush=True)
    if r is not None:
        print(f"     [!] ffmpeg failed:\n{r.stderr.strip()[-400:]}")
    return False


def transcode(inplace: bool = False, mode: str = "gpu") -> tuple:
    """Transcode every AV1 video in videos/ to H.264, frame-preserving.

    inplace=True overwrites the originals in videos/ (verified before replace) so
    the local Stage-1 detector can read them; inplace=False writes copies to
    videos_h264/. mode is 'gpu' (av1_cuvid+h264_nvenc), 'hwdec' (GPU decode +
    libx264), or 'cpu'. Returns (ok, bad) lists of filenames. Importable so
    stage1.py can drive it; main() just parses args and calls this."""
    if not SRC_DIR.is_dir():
        print(f"No {SRC_DIR}/ folder. Add videos or set DASH_RAW_DIR.")
        return [], []

    vids = sorted(p for p in SRC_DIR.iterdir()
                  if p.suffix.lower() in (".mp4", ".mkv", ".mov", ".webm"))
    av1 = [p for p in vids if (probe(p, "codec_name") or [""])[0] == "av1"]

    if not av1:
        print("No AV1 videos found — nothing to do.")
        return [], []

    out_desc = "in place (videos/)" if inplace else f"-> {OUT_DIR}/"
    eng = {"gpu":   "full GPU (av1_cuvid + h264_nvenc)",
           "hwdec": "GPU decode + CPU encode (libx264)",
           "cpu":   "CPU only (libx264)"}[mode]
    if not inplace:
        OUT_DIR.mkdir(exist_ok=True)
    print(f"Found {len(av1)} AV1 video(s). Transcoding to H.264  |  "
          f"output: {out_desc}  |  engine: {eng}\n")

    ok, bad = [], []
    for p in av1:
        src_frames = (probe(p, "nb_frames") or ["?"])[0]
        print(f"  {p.name}  ({src_frames} frames) ...", flush=True)

        # always write to a temp file first; only commit if frames match.
        tmp = p.with_name(p.stem + ".h264tmp.mp4")
        if not _encode(p, tmp, mode):
            tmp.unlink(missing_ok=True)
            bad.append(p.name)
            continue

        dst_frames = (probe(tmp, "nb_frames") or ["?"])[0]
        verified = src_frames.isdigit() and src_frames == dst_frames
        if not verified:
            print(f"     -> {dst_frames} frames  FRAME COUNT CHANGED / "
                  f"UNVERIFIABLE — keeping original, leaving temp for inspection: "
                  f"{tmp.name}")
            bad.append(p.name)
            continue

        if inplace:
            final = p.with_suffix(".mp4")     # H.264 in an mp4 container
            p.unlink()                        # drop the AV1 original (verified safe)
            os.replace(tmp, final)            # temp -> final name, atomic
            print(f"     -> replaced {final.name}  ({dst_frames} frames)  OK")
        else:
            final = OUT_DIR / (p.stem + ".mp4")
            os.replace(tmp, final)
            print(f"     -> {final.name}  ({dst_frames} frames)  OK")
        ok.append(p.name)

    print("\nDone.")
    print(f"  {len(ok)} transcoded with matching frame counts.")
    if bad:
        print(f"  [!] {len(bad)} failed or changed frame count "
              f"(originals kept): {bad}")
    if not inplace and ok:
        print(f"\nNext: in your Kaggle videos dataset, replace these files with "
              f"the {OUT_DIR}/ versions (same names), then re-run Stage 2.")
    return ok, bad


def main():
    ap = argparse.ArgumentParser(description="Transcode AV1 videos to H.264.")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite originals in videos/ (verified before replace)")
    ap.add_argument("--hwdec", action="store_true",
                    help="GPU decode + CPU libx264 encode (CRF 16)")
    ap.add_argument("--cpu", action="store_true",
                    help="force pure CPU libx264 (no GPU)")
    args = ap.parse_args()

    # full GPU is the default; --hwdec / --cpu step down from there.
    mode = "cpu" if args.cpu else ("hwdec" if args.hwdec else "gpu")
    transcode(inplace=args.inplace, mode=mode)


if __name__ == "__main__":
    main()
