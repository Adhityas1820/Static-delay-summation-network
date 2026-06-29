"""
features.py
-----------
Spatial front-end: turn each frame into a compact CNN feature vector.

A general-purpose ImageNet ResNet18 (frozen, fc head stripped) reads every
frame and emits a 512-d global-avg-pool vector. Weights are shared across time,
so a video becomes a sequence of compact vectors that the SDSNN then models
temporally. No optical flow, no raw pixels into the SDSNN.

A general encoder is used on purpose: its features stay sensitive to the dash
VFX. Set DASH_BACKBONE to a checkpoint path only if you want a fine-tuned
encoder (e.g. the map classifier) instead.

Output shape: [n_frames, FEAT_DIM], aligned with the dash detector's frame
indexing (every raw frame, in order).

Decoding: Stage 2 is video-decode-bound, not GPU-bound — ResNet18 inference is
cheap, so the GPU mostly idles while OpenCV decodes 1080p frames one at a time.
We therefore decode with `decord` when available: it reads frames in batches in
C++ and downscales to IMG_SIZE during decode (we throw away everything above
224px anyway), which is far faster and keeps the GPU fed. If decord is missing
or fails on a file, we fall back to the original frame-by-frame OpenCV path.
Both paths produce one row per decoded frame, in order — identical alignment.

torch/torchvision/decord are imported lazily inside the functions so that
Stage 1 (local dash detection + CSV review) runs with only OpenCV/NumPy.
"""

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .dash_counter import open_video, _progress_bar

N_FEATURES    = C.FEAT_DIM
FEATURE_NAMES = [f"cnn{i}" for i in range(N_FEATURES)]

_backbone = None
_device   = None
_mean     = None
_std      = None
_decoder_announced = False


def _get_backbone():
    global _backbone, _device, _mean, _std
    if _backbone is not None:
        return _backbone, _device

    import torch
    import torch.nn as nn
    from torchvision import models

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if C.BACKBONE_CKPT:
        ckpt_path = Path(C.BACKBONE_CKPT)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"DASH_BACKBONE set to '{ckpt_path}' but it does not exist.")
        ckpt    = torch.load(str(ckpt_path), map_location="cpu")
        classes = ckpt.get("classes", [])
        net = models.resnet18(weights=None)
        n_cls = len(classes) if classes else net.fc.out_features
        net.fc = nn.Linear(net.fc.in_features, n_cls)
        net.load_state_dict(ckpt["model_state_dict"])
        print(f"  backbone: fine-tuned ResNet18 ({n_cls} classes) <- {ckpt_path.name}")
    else:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print("  backbone: ImageNet ResNet18 (general-purpose, default)")

    net.fc = nn.Identity()                      # expose 512-d avgpool features
    net.eval().to(_device)
    _backbone = net
    _mean = torch.tensor(C.IMAGENET_MEAN, device=_device).view(1, 3, 1, 1)
    _std  = torch.tensor(C.IMAGENET_STD,  device=_device).view(1, 3, 1, 1)
    return _backbone, _device


def _normalize(t):
    """uint8 NHWC RGB tensor on any device -> normalized NCHW float on _device."""
    t = t.to(_device).float().div_(255.0)
    t = t.permute(0, 3, 1, 2)                            # NHWC -> NCHW
    return (t - _mean) / _std


def _normalize_nchw_uint8(t):
    """[B,C,H,W] uint8 RGB tensor (any device) -> resized, normalized NCHW float
    on _device. Used by the torchcodec path, which already returns NCHW. Resize
    is done in uint8 so the full-res copy stays small (no full-res float blowup),
    then cast + normalized."""
    from torchvision.transforms.v2 import functional as TF
    t = TF.resize(t.to(_device), [C.IMG_SIZE, C.IMG_SIZE], antialias=True)
    t = t.float().div_(255.0)
    return (t - _mean) / _std


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _preprocess(frames_bgr):
    """list of HxWx3 uint8 BGR (OpenCV path) -> normalized NCHW float tensor."""
    import torch
    arr = np.stack([cv2.resize(f, (C.IMG_SIZE, C.IMG_SIZE)) for f in frames_bgr])
    arr = np.ascontiguousarray(arr[:, :, :, ::-1])      # BGR -> RGB
    t = torch.from_numpy(arr)
    return _normalize(t)


def _announce(decoder: str):
    global _decoder_announced
    if not _decoder_announced:
        print(f"  decoder: {decoder}")
        _decoder_announced = True


def _stage_to_ram(video_path):
    """Copy the source video onto a RAM-backed filesystem (Linux tmpfs at
    /dev/shm) before decoding, so decord's many reads/seeks hit RAM instead of
    a slow mount — e.g. Kaggle's /kaggle/input FUSE layer or a network drive.

    We stage the *file* (tens-to-few-hundred MB), not the decoded frames
    (which would be many GB and could OOM). Returns (path_to_open,
    tmp_to_cleanup_or_None). No-op on Windows (no tmpfs) or on any error, in
    which case the original path is returned unchanged.
    """
    shm = Path("/dev/shm")
    try:
        if not shm.is_dir():
            return str(video_path), None          # Windows / no tmpfs
        size = Path(video_path).stat().st_size
        if size > 0.5 * shutil.disk_usage(shm).free:
            return str(video_path), None          # don't risk filling RAM
        fd, tmp = tempfile.mkstemp(suffix=Path(video_path).suffix, dir=str(shm))
        os.close(fd)
        shutil.copy2(str(video_path), tmp)
        print(f"  staged {Path(video_path).name} -> RAM ({size/1e6:.0f} MB)",
              flush=True)
        return tmp, tmp
    except Exception:
        return str(video_path), None


# ---------------------------------------------------------------------------
# torchcodec path (fastest): true NVDEC hardware decode straight onto the GPU
# ---------------------------------------------------------------------------
def _extract_torchcodec(video_path) -> np.ndarray:
    """Decode on the GPU's hardware video decoder (NVDEC) via torchcodec and
    keep the frames on the GPU the whole way through ResNet.

    Unlike decord — whose pip wheels are CPU-only, so its `decord.gpu(0)` path
    silently falls back to CPU decode — torchcodec hands back CUDA tensors
    decoded by NVDEC, so decode/resize/normalize/forward never round-trip
    through the CPU. This is the fix for the "decode is CPU-bound" bottleneck.

    Frames come back as [B, C, H, W] uint8 RGB on the backbone's CUDA device.
    We decode in FEATURE_BATCH-sized ranges so a long video never has all its
    frames resident at once (full-res frames on the GPU would OOM).
    """
    import torch
    from torchcodec.decoders import VideoDecoder

    net, _ = _get_backbone()                    # sets _device / _mean / _std
    dec = VideoDecoder(str(video_path), device=str(_device),
                       dimension_order="NCHW")
    _announce("torchcodec (GPU/NVDEC)")
    n = len(dec)

    feats = []
    with torch.no_grad():
        for start in range(0, n, C.FEATURE_BATCH):
            stop = min(start + C.FEATURE_BATCH, n)
            frames = dec[start:stop]            # [B, C, H, W] uint8 RGB on CUDA
            out = net(_normalize_nchw_uint8(frames)).cpu().numpy().astype(np.float32)
            feats.append(out)
            _progress_bar(stop, n, label="frames")

    return np.concatenate(feats) if feats else np.zeros((0, N_FEATURES), dtype=np.float32)


# ---------------------------------------------------------------------------
# decord path (fast): batched reads + decode-time downscale to IMG_SIZE
# ---------------------------------------------------------------------------
def _extract_decord(video_path) -> np.ndarray:
    import torch
    import decord
    from decord import VideoReader

    decord.bridge.set_bridge("torch")           # get_batch -> torch tensors

    # decode straight to IMG_SIZE (RGB); try GPU decode, fall back to CPU.
    try:
        gpu_ctx = decord.gpu(0)
        vr = VideoReader(str(video_path), ctx=gpu_ctx,
                         width=C.IMG_SIZE, height=C.IMG_SIZE)
        _announce("decord (GPU/NVDEC, downscaled)")
    except Exception:
        vr = VideoReader(str(video_path), ctx=decord.cpu(0),
                         width=C.IMG_SIZE, height=C.IMG_SIZE)
        _announce("decord (CPU, downscaled)")

    net, _ = _get_backbone()
    n = len(vr)
    feats = []
    with torch.no_grad():
        for start in range(0, n, C.FEATURE_BATCH):
            idx = list(range(start, min(start + C.FEATURE_BATCH, n)))
            batch = vr.get_batch(idx)           # [B, H, W, 3] uint8 RGB
            out = net(_normalize(batch)).cpu().numpy().astype(np.float32)
            feats.append(out)
            _progress_bar(min(start + C.FEATURE_BATCH, n), n, label="frames")

    return np.concatenate(feats) if feats else np.zeros((0, N_FEATURES), dtype=np.float32)


# ---------------------------------------------------------------------------
# OpenCV path (fallback): frame-by-frame decode
# ---------------------------------------------------------------------------
def _extract_opencv(video_path) -> np.ndarray:
    import torch
    _announce("OpenCV (frame-by-frame, fallback)")
    net, _ = _get_backbone()
    cap, tmp_path = open_video(video_path)
    if not cap.isOpened():
        cap.release()
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    feats, batch = [], []
    done = 0

    def flush():
        nonlocal done
        if batch:
            out = net(_preprocess(batch)).cpu().numpy().astype(np.float32)
            feats.append(out)
            done += len(batch)
            _progress_bar(min(done, total), total, label="frames")
            batch.clear()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(frame)
            if len(batch) >= C.FEATURE_BATCH:
                flush()
        flush()

    cap.release()
    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)

    return np.concatenate(feats) if feats else np.zeros((0, N_FEATURES), dtype=np.float32)


def extract_cnn_features(video_path) -> np.ndarray:
    """Returns float32 array [n_frames, FEAT_DIM]. Decoder priority:
    torchcodec (GPU/NVDEC, if a CUDA device is present) -> decord (CPU) ->
    OpenCV. All three decode every frame in order (same label alignment).

    The source file is first staged onto RAM (tmpfs) when available so decode
    isn't bottlenecked on slow storage; the staged copy is always cleaned up.
    """
    path, tmp = _stage_to_ram(video_path)
    try:
        # Preferred: true GPU decode (NVDEC) via torchcodec. Only attempted when
        # a CUDA device exists; any failure (not installed, no NVDEC, codec
        # unsupported by the HW decoder) falls through to decord -> OpenCV, so
        # this never regresses a CPU-only box.
        if _cuda_available():
            try:
                return _extract_torchcodec(path)
            except Exception as e:
                print(f"  torchcodec GPU decode unavailable on "
                      f"{Path(video_path).name} ({e}); trying decord/OpenCV.")
        try:
            import decord  # noqa: F401
        except Exception:
            return _extract_opencv(path)
        try:
            return _extract_decord(path)
        except Exception as e:
            print(f"  decord failed on {Path(video_path).name} ({e}); "
                  f"falling back to OpenCV.")
            return _extract_opencv(path)
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)
