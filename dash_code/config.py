"""
config.py
---------
Shared configuration for the SDSNN dash-detector data pipeline.

The bottom block (HUD geometry + thresholds) is copied verbatim from the
Marvel Rivals classifier's `dash_counter.py` — that is the contour/UI dash
detector we REUSE to get dash start points. Do not retune it here.

The top block adds what the frame-level SDSNN pipeline needs: interval
geometry (the fixed ~450 ms event prior), CNN front-end params, and paths.

All paths resolve relative to the project root by default but can be
overridden with environment variables, so train.py stays portable (Kaggle).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else default


# --- I/O paths -------------------------------------------------------------
RAW_VIDEO_DIR = _env_path("DASH_RAW_DIR",       PROJECT_ROOT / "videos")
PROCESSED_DIR = _env_path("DASH_PROCESSED_DIR", PROJECT_ROOT / "processed")
FEATURES_DIR  = PROCESSED_DIR / "features"          # per-video .npz lives here
LABELS_CSV    = _env_path("DASH_LABELS_CSV",    PROCESSED_DIR / "dash_intervals.csv")
REVIEW_DIR    = _env_path("DASH_REVIEW_DIR",    PROJECT_ROOT / "review")  # clips / overlay
MODELS_DIR    = _env_path("DASH_MODELS_DIR",    PROJECT_ROOT / "models")

CHECKPOINT_DIR = _env_path("DASH_CKPT_DIR", PROJECT_ROOT / "checkpoints")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# --- frame rate / timing ---------------------------------------------------
FPS          = 60                 # Marvel Rivals clips are 1920x1080 @ 60 fps
MS_PER_FRAME = 1000.0 / FPS       # 16.667 ms

# --- dash event geometry (the inference prior) -----------------------------
# A dash is a fixed ~450 ms event (~27 frames). Around each detected START we
# build a labelled interval:
#   start = point - 50 ms  (3 frames before)
#   end   = point + 430 ms (26 frames after)
# => ~480 ms / ~29-frame window. Matches the expected detection width.
PRE_DASH_MS  = 50.0
POST_DASH_MS = 430.0

# --- completion-spike target (the temporal reframe, 2026-06-22) -------------
# The SDSNN target is no longer "1 across the whole dash". Instead it is a soft
# Gaussian bump that PEAKS when the dash has fully UNFOLDED (its falling edge,
# ~one dash-length after the detected start). Two reasons:
#   * #peaks == #dashes, so the model finally emits a COUNT (1-D NMS over the
#     probability track), the thing the parent counter actually wants.
#   * "a dash just completed" is intrinsically temporal — the answer at frame t
#     depends on frames t-29..t — so it is NOT per-frame-decodable and genuinely
#     needs the SDSNN's delay window (a linear probe physically can't represent
#     it). The old per-frame "dash vs non-dash" target was per-frame-decodable,
#     which is why a 2-line smoothing rule tied the SDSNN there. See decisions.
# COMPLETION_MS is measured from the dash START; GAUSS_SIGMA is the bump width
# in frames. A single hot frame won't train at <1% positives — the soft bump
# gives the loss a gradient and a single clear peak per dash to count.
COMPLETION_MS = 450.0   # dash fully unfolded (~27 frames after the start)
GAUSS_SIGMA   = 3.0     # completion-bump width in frames (FWHM ~7f)

# --- completion peak decoding + loss (Stage 3 train/eval, Stage 4 counting) --
# The probability track is turned into a dash COUNT by 1-D non-max suppression:
# take the highest frame above PEAK_THRESHOLD, count it, suppress everything
# within PEAK_MIN_DIST, repeat. MIN_DIST sits below the ~18f rearm / ~22f
# observed chained-dash gap so chained completions stay separable.
PEAK_MIN_DIST     = 12    # min frames between two counted peaks (NMS)
PEAK_MATCH_TOL    = 8     # a predicted peak matches a GT completion within +/- this
PEAK_THRESHOLD    = 0.5   # default NMS confidence cutoff (swept during training)
HEATMAP_POS_WEIGHT = 60.0 # up-weight the sparse peak region in the heatmap MSE

# --- CNN front-end (spatial features) --------------------------------------
# Two-stage architecture: a ResNet18 turns each frame into a compact 512-d
# feature vector (the spatial half), then the SDSNN models the ~450 ms temporal
# pattern over that sequence (the temporal half). The backbone is run FROZEN
# with its fc head stripped to expose the 512-d global-avg-pool features.
#
# Default backbone = general-purpose ImageNet ResNet18. We deliberately do NOT
# default to the Marvel Rivals map classifier: its features are fine-tuned to
# tell environments apart and are largely invariant to transient VFX, so they
# would be blind to the dash effect we need to detect. A general encoder keeps
# broad, VFX-sensitive features.
#
# To use a fine-tuned checkpoint instead, set DASH_BACKBONE to its path.
BACKBONE_CKPT = os.environ.get("DASH_BACKBONE", "")   # "" -> ImageNet ResNet18
FEAT_DIM      = 512               # resnet18 avgpool features
IMG_SIZE      = 224
FEATURE_BATCH = 64                # frames per forward pass
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- SDSNN training (Phase 2) ----------------------------------------------
WINDOW     = 128        # frames per training sequence (must exceed MAX_DELAY)
STRIDE     = 64         # window hop over each video
HIDDEN     = 64         # delay-layer width
MAX_DELAY  = 32         # delay slots; spans the ~27-frame dash event
N_EPOCHS   = 30
BATCH_SIZE = 32
LR         = 1e-3
VAL_FRAC   = 0.15       # fraction of VIDEOS held out for validation
SEED       = 42
THRESHOLD  = 0.5        # per-frame prob -> dash for metrics
# loss: "bce" (pos-weighted) or "focal"
LOSS        = "bce"
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# --- review annotation video -----------------------------------------------
# Annotated overlay clips are downscaled to this size for fast, small review
# output (the source is 1920x1080; this keeps the writes cheap for many clips).
REVIEW_W = 960
REVIEW_H = 540

# ===========================================================================
# REUSED DASH DETECTOR — copied verbatim from
# MarvelRivalsClassifier/dash_counter.py. This is the exact contour/UI logic;
# do NOT invent a new detector or retune these constants.
# ===========================================================================
PROCESS_FPS = 60

SLOT2_SEARCH = (1575, 1625, 965, 1000)
SLOT3_SEARCH = (1500, 1550, 965, 1000)
SLOT2_LABEL  = (1575, 1625, 1030, 1050)
SLOT3_LABEL  = (1500, 1550, 1030, 1050)
SLOT_DETECT_FRAMES = 240

# Contours are now split into models/clips/ and models/videos/ and selected
# per source by duration at runtime (see dash_counter.contour_paths_for).
# These defaults point at the short-clip folder for any code that wants a
# single static path.
RIGHT_CONTOUR_PATH = str(MODELS_DIR / "clips" / "slot_x_contour_right.npy")
LEFT_CONTOUR_PATH  = str(MODELS_DIR / "clips" / "slot_x_contour_left.npy")

WHITE_THRESH       = 200
WHITE_RATIO_THRESH = 0.95
LABEL_GREY_THRESH  = 110
ZOOM_LOW_THRESH    = 0.5
OFF_FRAMES         = 3
DASH_REARM_SECS    = 0.3
