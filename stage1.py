"""
stage1.py  —  STAGE 1 (local), end to end
==========================================
One entry point for everything that happens on this machine before the data
goes to Kaggle. It runs, in order:

  1. transcode  — re-encode any AV1 videos in videos/ to H.264 IN PLACE, so the
                  local dash detector (and later decoders) can actually read them.
                  AV1 silently decodes to 0 frames otherwise. Frame-preserving,
                  so existing hand-reviewed labels stay aligned. GPU by default.
  2. labels     — the "first part" of process_data: run the reused contour dash
                  detector over every video -> dash intervals -> write
                  processed/dash_intervals.csv, then render review overlays.
  3. dash-videos — compress every clip to 360p H.264 (frame-preserving, same
                  filenames) and copy the CSV into ./dash-videos/, so that folder
                  is ready to upload straight to Kaggle. Incremental: unchanged
                  clips are skipped on a re-run (only the CSV re-copies).

So one `python stage1.py` leaves both processed/dash_intervals.csv (review it!)
and a ready-to-upload dash-videos/ folder. Stage 2 (ResNet18 features) and Stage 3
(train the SDSNN) run in the Kaggle notebook; Stage 4 is stage4.py (local scoring).

Usage (from the project root):
    python stage1.py                 # transcode + labels + overlays + dash-videos/
    python stage1.py --no-overlay    # skip overlay videos (faster)
    python stage1.py --clips         # also export per-interval clips
    python stage1.py --redetect      # ignore the detection cache
    python stage1.py --skip-transcode  # skip the AV1 re-encode
    python stage1.py --skip-videos     # skip building dash-videos/ (labels only)
    python stage1.py --force-videos    # re-compress every clip (ignore the cache)
    python stage1.py --hwdec | --cpu   # step transcode off full-GPU
"""

import argparse
import multiprocessing as mp
import os
import sys

# stage1.py lives at the repo root, so its own directory is the root — the home
# of the dash_code/ package. Put it on sys.path so `from helper... import` resolves
# no matter the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dash_code import config as C
from dash_code import process_data as pd
from dash_code import transcode_av1 as tc
from dash_code import make_dash_videos as mdv


def main():
    ap = argparse.ArgumentParser(
        description="Stage 1 (local): transcode AV1 -> H.264, then detect dashes "
                    "and write the review CSV.")
    # labels-stage pass-throughs (see dash_code/process_data.py)
    ap.add_argument("--clips", action="store_true",
                    help="also export per-interval review clips")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip the annotated overlay videos (faster)")
    ap.add_argument("--redetect", action="store_true",
                    help="ignore the detection cache and re-detect every video")
    # transcode controls
    ap.add_argument("--skip-transcode", action="store_true",
                    help="skip the AV1->H.264 step (do labels only)")
    ap.add_argument("--hwdec", action="store_true",
                    help="transcode: GPU decode + CPU libx264 encode")
    ap.add_argument("--cpu", action="store_true",
                    help="transcode: force pure CPU libx264 (no GPU)")
    # dash-videos build controls
    ap.add_argument("--skip-videos", action="store_true",
                    help="skip building the dash-videos/ upload folder")
    ap.add_argument("--force-videos", action="store_true",
                    help="re-compress every clip into dash-videos/ (ignore the cache)")
    ap.add_argument("--video-height", type=int, default=360,
                    help="dash-videos target height (default 360)")
    ap.add_argument("--video-crf", type=int, default=28,
                    help="dash-videos libx264 CRF (default 28)")
    args = ap.parse_args()

    # --- 1. transcode AV1 -> H.264 in place (GPU by default) ---------------
    if not args.skip_transcode:
        print("=== STAGE 1a: transcode AV1 videos -> H.264 (in place) ===")
        mode = "cpu" if args.cpu else ("hwdec" if args.hwdec else "gpu")
        tc.transcode(inplace=True, mode=mode)
        print()

    # --- 2. labels: detect dashes -> intervals -> review CSV ---------------
    videos = pd.list_videos()
    if not videos:
        print(f"No videos in {C.RAW_VIDEO_DIR}. Add clips (1920x1080 @ 60 fps) "
              f"or set DASH_RAW_DIR.")
        return
    print(f"=== STAGE 1b: detect dashes in {len(videos)} video(s) -> review CSV ===")
    pd.stage_labels(videos, args.clips, not args.no_overlay, redetect=args.redetect)

    # --- 3. compress -> dash-videos/ (CSV + small clips, ready for Kaggle) --
    if not args.skip_videos:
        print("\n=== STAGE 1c: build dash-videos/ (compress clips + copy CSV) ===")
        mdv.build(height=args.video_height, crf=args.video_crf,
                  hwaccel=not args.cpu, force=args.force_videos)
        print(f"\ndash-videos/ ready -> upload it as the `dash-videos` Kaggle dataset.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
