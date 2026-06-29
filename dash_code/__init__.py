"""
dash_code — the SDSNN dash-detector pipeline as ONE flat package.

All modules live directly here (no stage subfolders), so the local layout matches
the flat `dash-code` Kaggle dataset 1:1 — the notebook just copies every *.py in
and runs `python -m dash_code.<module>`. No subpackage routing, nothing to keep
in sync.

Stage 1 (local):  transcode_av1, dash_counter, process_data --stage labels  (-> dash_intervals.csv)
Stage 2 (Kaggle): process_data --stage features  (-> processed/features/*.npz, ResNet18 + Gaussian completion labels)
Stage 3 (Kaggle): train  (SDSNN heatmap regression, NMS-threshold sweep)
Stage 4 (local):  ../stage4.py  (NMS peak counting -> dashes per video)

Shared: config (paths/constants), io_utils (CSV/.npz), labels (completion target),
peaks (NMS counting, shared by train + stage4).
"""
