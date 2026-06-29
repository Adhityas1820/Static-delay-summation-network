"""
dataset.py
----------
Load the per-video .npz arrays from Phase 1 and serve them to the SDSNN as
fixed-length windows.

Split is BY VIDEO (not by window) so no frames from a validation clip leak into
training. Each item is (x, y, mask): a [WINDOW, F] input, [WINDOW] labels, and
a [WINDOW] mask that is 0 on right-pad frames (so short clips don't inject fake
negatives into the loss/metrics).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C


def list_processed():
    d = C.FEATURES_DIR
    return sorted(d.glob("*.npz")) if d.exists() else []


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["features"].astype(np.float32), z["labels"].astype(np.float32)


def split_videos(files, val_frac=C.VAL_FRAC, seed=C.SEED):
    # Hardcoded hold-out test set per user request
    hold_out_names = [
        "training video 08", "training video 18", "training video 19",
        "training video 25", "training video 28", "training video 29",
        "training video 38", "training video 54", "training video 64"
    ]
    
    train = []
    valf = []
    for f in files:
        if any(f.name.startswith(name) for name in hold_out_names):
            valf.append(f)
        else:
            train.append(f)
    return train, valf


class WindowDataset(Dataset):
    def __init__(self, files, window=C.WINDOW, stride=C.STRIDE):
        self.window = window
        self.arrays = []     # [(feat, lab), ...]
        self.index  = []     # [(array_idx, start), ...]
        for f in files:
            feat, lab = load_npz(f)
            n = feat.shape[0]
            if n == 0:
                continue
            ai = len(self.arrays)
            self.arrays.append((feat, lab))
            if n <= window:
                self.index.append((ai, 0))
            else:
                starts = list(range(0, n - window + 1, stride))
                if starts[-1] != n - window:        # make sure the tail is covered
                    starts.append(n - window)
                self.index.extend((ai, s) for s in starts)

    def pos_neg_counts(self):
        pos = neg = 0.0
        for _, lab in self.arrays:
            pos += float((lab > 0.5).sum())
            neg += float((lab <= 0.5).sum())
        return pos, neg

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ai, s = self.index[i]
        feat, lab = self.arrays[ai]
        x = feat[s:s + self.window]
        y = lab[s:s + self.window]
        m = np.ones(x.shape[0], dtype=np.float32)
        if x.shape[0] < self.window:
            pad = self.window - x.shape[0]
            x = np.pad(x, ((0, pad), (0, 0)))
            y = np.pad(y, (0, pad))
            m = np.pad(m, (0, pad))
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)
