"""
flow_probe.py  —  Does optical flow carry signal where the appearance model MISSES?
===================================================================================
The question (before building a two-stream motion front-end): the residual Stage-4
misses are "ult / occlusion" dashes the frozen-ResNet appearance model can't see.
If we add an optical-flow stream, it only helps if THOSE missed dashes actually have
a motion signature. If the missed dashes are flat in flow too (truly occluded), the
flow stream is blind there as well and the whole two-stream build won't fix them.

What this does, per test clip:
  1. Load the cached 512-d ResNet features (predictions/cache/<stem>_features.npy).
  2. Run the SAME trained checkpoint -> per-frame completion prob -> NMS peaks
     (checkpoint's baked threshold), exactly like Stage 4.
  3. Match peaks to the cached GT completion centers -> label each GT dash HIT or
     MISSED (the model's false-negatives are the dashes of interest).
  4. Compute an optical-flow MAGNITUDE track only in a short window around each GT
     dash (the dash unfolds in the ~27 frames before its completion frame) plus a
     sample of non-dash baseline frames.
  5. Score each dash:  motion_ratio = peak_flow_in_window / clip_baseline_flow.

Read it:
  * MISSED dashes have motion_ratio ~ HIT dashes (both >> 1)  -> motion signal IS
    present at the misses; appearance model just isn't using it -> two-stream worth
    building.
  * MISSED dashes have motion_ratio ~ 1 (flat, == baseline)   -> flow is blind there
    too (occlusion); two-stream won't recover them -> save the build.

Flow: torchvision RAFT-small on GPU (fast, high quality); falls back to Farneback
(CPU) if the RAFT weights can't be fetched. Frames are read with decord, downscaled
during decode — same per-frame ordering as feature extraction, so indices line up.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root on path
from dash_code import config as C
from dash_code import peaks as pk
from dash_code.dnn import DNN
from dash_code.labels import fmt_ts

ROOT      = C.PROJECT_ROOT
CACHE     = ROOT / "predictions" / "cache"
TEST_DIR  = ROOT / "test set"
CKPT      = ROOT / "checkpoints" / "sdsnn.pt"
OUT_DIR   = ROOT / "Claude_experiments" / "output"

FLOW_H, FLOW_W = 256, 256        # RAFT needs dims divisible by 8; magnitude probe
PRE_WIN   = 31                   # frames before completion to scan (dash ~27f long)
POST_WIN  = 3                    # a few frames after completion
N_BASE    = 20                   # baseline (non-dash) sample windows per clip
BASE_GAP  = 45                   # baseline frames must be this far from any GT dash

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- model
def load_model():
    ck = torch.load(str(CKPT), map_location=DEVICE)
    m = DNN(in_dim=ck.get("in_dim", C.FEAT_DIM),
            hidden=ck.get("hidden", 64), layers=ck.get("layers", 4)).to(DEVICE)
    m.load_state_dict(ck["model_state"]); m.eval()
    thr = ck.get("threshold", C.PEAK_THRESHOLD)
    mind = ck.get("peak_min_dist", C.PEAK_MIN_DIST)
    tol = ck.get("peak_match_tol", C.PEAK_MATCH_TOL)
    return m, thr, mind, tol


def model_prob(m, feats):
    x = torch.from_numpy(feats[None]).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(m(x)).squeeze(0).cpu().numpy()


# ----------------------------------------------------------------------------- flow
class FlowEngine:
    """Mean optical-flow magnitude between consecutive frames. RAFT on GPU when the
    weights load, else Farneback on CPU. Either way: input two HxWx3 uint8 RGB frames
    -> scalar mean |flow|."""
    def __init__(self):
        self.kind = None
        self.model = None
        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            w = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=w, progress=False).to(DEVICE).eval()
            self.tf = w.transforms()
            self.kind = "raft"
        except Exception as e:
            print(f"  RAFT unavailable ({type(e).__name__}: {e}); using Farneback (CPU).")
            import cv2  # noqa
            self.kind = "farneback"

    def batch_mag(self, frames):
        """frames: uint8 [N,H,W,3] RGB in order. Returns mean |flow| for each
        consecutive pair -> array length N-1."""
        if self.kind == "raft":
            return self._raft(frames)
        return self._farneback(frames)

    def _raft(self, frames):
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()      # [N,3,H,W]
        img1 = t[:-1]; img2 = t[1:]
        mags = []
        bs = 4
        with torch.no_grad():
            for s in range(0, img1.shape[0], bs):
                a = img1[s:s+bs].to(DEVICE); b = img2[s:s+bs].to(DEVICE)
                a, b = self.tf(a, b)
                flow = self.model(a, b)[-1]                            # [b,2,H,W]
                mag = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2).mean(dim=(1, 2))
                mags.append(mag.cpu().numpy())
        return np.concatenate(mags) if mags else np.zeros(0, np.float32)

    def _farneback(self, frames):
        import cv2
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        out = []
        for i in range(1, len(grays)):
            fl = cv2.calcOpticalFlowFarneback(grays[i-1], grays[i], None,
                                              0.5, 3, 15, 3, 5, 1.2, 0)
            out.append(float(np.sqrt(fl[..., 0]**2 + fl[..., 1]**2).mean()))
        return np.asarray(out, np.float32)


# ------------------------------------------------------------------------ per clip
def read_frames(vr, lo, hi):
    """decord read of frame indices [lo, hi) (clamped), downscaled, RGB uint8."""
    n = len(vr)
    idx = [min(max(i, 0), n - 1) for i in range(lo, hi)]
    return vr.get_batch(idx).asnumpy()


def window_motion(flow_eng, vr, center, T):
    """Peak mean-|flow| over the dash window [center-PRE_WIN, center+POST_WIN]."""
    lo = center - PRE_WIN - 1
    hi = min(T, center + POST_WIN + 1)
    frames = read_frames(vr, lo, hi)
    mags = flow_eng.batch_mag(frames)
    return float(mags.max()) if mags.size else float("nan")


def baseline_motion(flow_eng, vr, gt, T, rng):
    """Median mean-|flow| over N_BASE random short windows >= BASE_GAP from any dash."""
    forbidden = np.zeros(T, dtype=bool)
    for c in gt:
        lo = max(0, c - PRE_WIN - BASE_GAP); hi = min(T, c + BASE_GAP)
        forbidden[lo:hi] = True
    ok = np.where(~forbidden)[0]
    ok = ok[(ok > 2) & (ok < T - 2)]
    if ok.size == 0:
        return float("nan")
    picks = rng.choice(ok, size=min(N_BASE, ok.size), replace=False)
    vals = []
    for p in picks:
        frames = read_frames(vr, p - 2, p + 2)
        mags = flow_eng.batch_mag(frames)
        if mags.size:
            vals.append(float(mags.max()))
    return float(np.median(vals)) if vals else float("nan")


def main():
    import decord
    decord.bridge.set_bridge("native")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m, thr, mind, tol = load_model()
    flow_eng = FlowEngine()
    print(f"device: {DEVICE} | flow: {flow_eng.kind} | thr {thr} min-dist {mind} tol {tol}\n")

    videos = sorted(f for f in TEST_DIR.iterdir()
                    if f.suffix.lower() in C.VIDEO_EXTENSIONS)
    rng = np.random.default_rng(42)
    rows = []

    for v in videos:
        feat_p = CACHE / f"{v.stem}_features.npy"
        gt_p   = CACHE / f"{v.stem}_gt.json"
        if not feat_p.exists() or not gt_p.exists():
            print(f"  [skip] {v.name}: no cached features/gt"); continue
        feats = np.load(feat_p)
        T = feats.shape[0]
        gt = sorted(int(c) for c in json.load(open(gt_p)) if 0 <= int(c) < T)
        if not gt:
            print(f"  [skip] {v.name}: 0 GT dashes"); continue

        prob = model_prob(m, feats)
        pred = pk.nms_peaks(prob, thr, mind)
        # which GT centers are matched (HIT) vs not (MISSED) — same greedy match
        used = [False] * len(gt)
        hit = [False] * len(gt)
        for p in sorted(pred):
            best, bd = -1, tol + 1
            for j, g in enumerate(gt):
                if used[j]:
                    continue
                d = abs(p - g)
                if d <= tol and d < bd:
                    bd, best = d, j
            if best >= 0:
                used[best] = True; hit[best] = True

        try:
            vr = decord.VideoReader(str(v), ctx=decord.cpu(0),
                                    width=FLOW_W, height=FLOW_H)
        except Exception as e:
            print(f"  [skip] {v.name}: decord open failed ({e})"); continue
        Tv = min(T, len(vr))

        base = baseline_motion(flow_eng, vr, gt, Tv, rng)
        n_miss = sum(1 for h in hit if not h)
        print(f"  {v.name[:34]:<34} dashes {len(gt):>3}  missed {n_miss}  "
              f"baseline|flow| {base:.3f}")
        for j, c in enumerate(gt):
            if c >= Tv:
                continue
            mot = window_motion(flow_eng, vr, c, Tv)
            ratio = mot / base if base and base > 1e-6 else float("nan")
            p_at = float(prob[c]) if c < len(prob) else float("nan")
            status = "HIT" if hit[j] else "MISS"
            rows.append(dict(video=v.name, completion_frame=c,
                             time=fmt_ts(c * C.MS_PER_FRAME), status=status,
                             prob=round(p_at, 3), peak_flow=round(mot, 4),
                             baseline_flow=round(base, 4), motion_ratio=round(ratio, 3)))
            if status == "MISS":
                print(f"      MISS @ {fmt_ts(c*C.MS_PER_FRAME):>9}  prob {p_at:.3f}  "
                      f"flow {mot:.3f}  ratio {ratio:.2f}x")

    # ---- write + summarize
    csv_p = OUT_DIR / "flow_probe.csv"
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    hit_r  = np.array([r["motion_ratio"] for r in rows
                       if r["status"] == "HIT"  and np.isfinite(r["motion_ratio"])])
    miss_r = np.array([r["motion_ratio"] for r in rows
                       if r["status"] == "MISS" and np.isfinite(r["motion_ratio"])])

    def stat(a):
        return (f"n={a.size}  median {np.median(a):.2f}x  mean {a.mean():.2f}x  "
                f"p25 {np.percentile(a,25):.2f}x  p75 {np.percentile(a,75):.2f}x") if a.size else "n=0"

    print("\n" + "=" * 70)
    print("MOTION RATIO  (peak |flow| in dash window  /  clip non-dash baseline)")
    print("=" * 70)
    print(f"  HIT  dashes : {stat(hit_r)}")
    print(f"  MISS dashes : {stat(miss_r)}")
    print("-" * 70)
    if miss_r.size and hit_r.size:
        print("Reading: MISS ratio ~ HIT ratio (both >>1)  -> motion present at misses; "
              "build two-stream.")
        print("         MISS ratio ~ 1 (flat)              -> flow blind at misses too; "
              "two-stream won't fix them.")
    print(f"\nCSV -> {csv_p}")


if __name__ == "__main__":
    main()
