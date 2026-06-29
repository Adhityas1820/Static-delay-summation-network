"""
combo_distribution.py
---------------------
How common are back-to-back dash COMBOS in the labelled data?

Reads the reviewed label CSV (processed/dash_intervals.csv), groups dashes by
video, and chains consecutive dashes into combos using the SAME rule as the
Marvel Rivals classifier (dash_code/dash_counter.py):

    a dash continues the current combo while
        (t - combo_start) <= 0.45 * (n - 1) + 0.275   [seconds]
    where combo_start = time of the FIRST dash in the combo, n = the would-be
    combo length. A dash is ~450 ms, so this lets each next dash land within
    roughly a dash-length of the previous (with a little slack).

Every combo is bucketed by length into Single / Double / Triple / Quad / Penta
(5+), counted across all videos, and drawn as a bar chart. Outputs go to
Claude_experiments/output/.
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "processed" / "dash_intervals.csv"
OUT_DIR  = ROOT / "Claude_experiments" / "output"

# --- combo rule (verbatim from dash_counter.py) ----------------------------
DASH_MS    = 450.0
WIN_STEP_S = 0.45      # 0.45 * (n-1)
WIN_SLACK_S = 0.275    # + 0.275
CATS = ["Single", "Double", "Triple", "Quad", "Penta"]   # Penta = 5 or more


def combo_lengths(times_sec):
    """Sorted dash start times (s) for ONE video -> list of combo lengths."""
    lengths = []
    combo_start = None
    count = 0
    for t in times_sec:
        if combo_start is None:
            combo_start, count = t, 1
        else:
            new = count + 1
            if (t - combo_start) <= WIN_STEP_S * (new - 1) + WIN_SLACK_S:
                count = new                       # extends the combo
            else:
                lengths.append(count)             # combo broke -> record it
                combo_start, count = t, 1
    if count > 0:
        lengths.append(count)
    return lengths


def bucket(n):
    return CATS[min(n, 5) - 1]                     # 1->Single ... 5+->Penta


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))

    by_video = {}
    for r in rows:
        try:
            by_video.setdefault(r["video"], []).append(float(r["dash_start_ms"]) / 1000.0)
        except (KeyError, ValueError):
            continue

    cat_counts = Counter()
    raw_len_counts = Counter()      # exact lengths (so we can see 6x+, if any)
    total_dashes = 0
    for video, times in by_video.items():
        for L in combo_lengths(sorted(times)):
            cat_counts[bucket(L)] += 1
            raw_len_counts[L] += 1
            total_dashes += L

    counts = [cat_counts.get(c, 0) for c in CATS]
    n_combos = sum(counts)
    most = CATS[counts.index(max(counts))]

    # --- bar chart ---------------------------------------------------------
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CATS, counts, color=colors, edgecolor="black", linewidth=0.6)
    for b, c in zip(bars, counts):
        pct = 100 * c / n_combos if n_combos else 0
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n({pct:.0f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Dash combos by length  ({total_dashes} dashes, "
                 f"{len(by_video)} videos)\nmost common: {most}", fontsize=12)
    ax.set_xlabel("Combo length (consecutive dashes)")
    ax.set_ylabel("Number of combos")
    ax.set_ylim(0, max(counts) * 1.18 if counts else 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png = OUT_DIR / "combo_distribution.png"
    fig.savefig(png, dpi=130)

    # --- text + csv summary ------------------------------------------------
    summary = OUT_DIR / "combo_distribution.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "combos", "pct_of_combos", "dashes_in_category"])
        for c, n in zip(CATS, counts):
            dashes = sum(L * raw_len_counts.get(L, 0)
                         for L in (range(5, max(raw_len_counts, default=5) + 1) if c == "Penta"
                                   else [CATS.index(c) + 1]))
            w.writerow([c, n, f"{100*n/n_combos:.1f}" if n_combos else 0, dashes])

    print(f"CSV   : {CSV_PATH}")
    print(f"videos: {len(by_video)}  |  dashes: {total_dashes}  |  combos: {n_combos}")
    print("\ncombo distribution:")
    for c, n in zip(CATS, counts):
        bar = "#" * round(40 * n / max(counts)) if counts and max(counts) else ""
        print(f"  {c:7s} {n:5d}  ({100*n/n_combos:4.1f}%)  {bar}")
    longest = max(raw_len_counts) if raw_len_counts else 0
    if longest > 5:
        over = sum(v for k, v in raw_len_counts.items() if k > 5)
        print(f"\n  note: {over} combo(s) longer than 5 (up to {longest}x) bucketed into Penta")
    print(f"\nMOST COMMON: {most}")
    print(f"\nchart -> {png}")
    print(f"table -> {summary}")


if __name__ == "__main__":
    main()
