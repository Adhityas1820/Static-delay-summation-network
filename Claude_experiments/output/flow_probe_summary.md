# Optical-flow probe — does motion carry signal where appearance MISSES?

Run: `python Claude_experiments/flow_probe.py`  (RAFT-small on GPU, decord decode)
Data: all 17 test clips with cached features; checkpoint `checkpoints/sdsnn.pt` @ thr 0.70.

## Question
Before building a two-stream motion front-end: the residual Stage-4 misses are the
"ult/occlusion" dashes. Does optical flow have signal at THOSE missed dashes, or are
they flat in flow too (truly occluded → flow can't help either)?

## Method
Per clip: run the trained model → NMS peaks → match to GT completion centers → label
each GT dash HIT or MISSED. Measure peak mean-|flow| (RAFT) in the dash window
[c-31, c+3] and divide by the clip's non-dash baseline (median of 20 random windows
≥45f from any dash). motion_ratio = peak_flow / baseline.

## Result
|            |  n  | median ratio | mean |
|------------|-----|--------------|------|
| HIT dashes | 169 | 2.24x        | 2.48x |
| MISS dashes|   8 | 2.38x        | 2.53x |

**Every one of the 8 misses is ≥1.76x baseline; missed dashes are motion-wise
indistinguishable from hit dashes.** None are flat/occluded.

Strongest evidence (appearance near-zero, flow strong — complementary, not redundant):
- training-70 @0:04:950 — prob 0.159, flow 2.35x
- training-70 @5:53:567 — prob 0.148, flow 2.40x
- luna @0:15:800       — prob 0.684 (just under 0.70), flow 3.46x
- purple-panther @0:14 — prob 0.602, flow 3.14x

## Verdict
Motion signal is present at 100% of the misses → **two-stream (appearance + flow) is
worth building.** Contrast with the purple experiment, which died because purple was
redundant with RGB; here flow lights up exactly where appearance is blind.

## Caveats
1. dash-vs-baseline ratio is inflated by window-length asymmetry (34f scan vs 4f); the
   HIT-vs-MISS comparison uses identical windows and is clean.
2. Shows motion is PRESENT at dashes, not yet that it's DISCRIMINATIVE vs other fast
   events (camera pans, other ults). Next step = the flow analogue of a linear probe:
   can flow features separate dash from non-dash motion.
