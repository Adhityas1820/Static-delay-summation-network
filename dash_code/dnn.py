"""
sdsnn.py
--------
The Static Delay-Summation Network (SDSNN) as a frame-level dash detector.

DelayLayer is copied VERBATIM from the project's SDSNN.py — the architecture
is reused, not redesigned. Each connection learns (via a softmax over K delay
slots) WHERE in the recent past to read from; a causal conv1d implements it.
There is no recurrence, so anything it solves, the delays solved.

The SDSNN module is a stack of delay layers (count = `layers`) then a linear
readout, parameterised for a variable input dimension and emitting a per-frame
logit (one dash score per timestep). The hidden activation is GELU (smooth, no
dead-neuron failure mode like ReLU, and unlike tanh it doesn't saturate the
magnitude of a sharp dash burst).

`residual` adds skip connections between same-width layers and `norm` adds a
LayerNorm after each layer — both off by default (so it matches the original
2-layer net), but turning them on is what makes a DEEPER stack trainable:
without them, gradients to the early layers vanish as depth grows (watch the
per-layer grad norms that train.py prints).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Fixed Delay Layer --------------------------------------------------------
class FixedDelayLayer(nn.Module):
    def __init__(self, c_in, c_out, dilation=1):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out)
        self.dilation = dilation

    def forward(self, x):
        # x: [B, T, c_in]
        h = self.linear(x)
        a = F.relu(h)
        
        # Apply delays
        B, T, C = a.shape
        groups = 4
        group_size = C // groups
        
        max_delay = 3 * self.dilation
        # pad only time dimension: [B, T+max_delay, C]
        a_padded = F.pad(a, (0, 0, max_delay, 0)) 
        
        out = torch.empty_like(a)
        for i in range(4):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < 3 else C
            delay = i * self.dilation
            
            # For delay d, we want time t to take value from time t-d.
            # In padded array, t-d is at index (t - d + max_delay)
            # So the sequence starts at (max_delay - delay) and ends at (max_delay - delay + T)
            start_t = max_delay - delay
            out[:, :, start_idx:end_idx] = a_padded[:, start_t : start_t + T, start_idx:end_idx]
            
        return out


# --- frame-level detector head --------------------------------------------
class DNN(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=4, **kwargs):
        super().__init__()
        # Ignore legacy kwargs like max_delay, residual, norm for compatibility with old train.py loading logic if any
        self.layers = nn.ModuleList()
        c_in = in_dim
        for i in range(layers):
            # Base-2 dilation: 1, 2, 4, 8...
            dilation = 2 ** i
            self.layers.append(FixedDelayLayer(c_in, hidden, dilation=dilation))
            c_in = hidden
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x):                          # x: [B, T, in_dim]
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.readout(h).squeeze(-1)         # logits [B, T]


# --- losses for the rare-positive frame problem ----------------------------
def focal_loss(logits, targets, alpha=0.25, gamma=2.0, mask=None):
    """Binary focal loss on per-frame logits. mask: 1 for real frames."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p  = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def weighted_bce(logits, targets, pos_weight, mask=None):
    """Pos-weighted BCE on per-frame logits. pos_weight: scalar tensor."""
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none")
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def heatmap_loss(logits, targets, mask=None, pos_weight=60.0):
    """Heatmap regression for the soft completion target (the temporal reframe).

    sigmoid(logits) is regressed to the [0,1] Gaussian target with a WEIGHTED
    MSE: w = 1 + pos_weight * target. The weighting up-weights the rare peak
    region so the net can't trivially win by predicting ~0 everywhere (peaks are
    <1% of frames). Unlike BCE on a hard 0/1 label, this gives a smooth gradient
    that pulls the output up toward the completion frame and back down — which is
    what makes the net emit a clean, countable bump per dash."""
    p = torch.sigmoid(logits)
    w = 1.0 + pos_weight * targets
    loss = w * (p - targets) ** 2
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()
