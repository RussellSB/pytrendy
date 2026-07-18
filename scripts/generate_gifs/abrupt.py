#!/usr/bin/env python3
"""Generate Abrupt.gif programmatically.

Recreates the animated GIF showing PyTrendy's abrupt trend detection,
with and without padding, in a two-cycle animation.

Usage:
    python scripts/generate_gifs/abrupt.py

Output:
    plots/Abrupt.gif
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytrendy as pt
from scripts.generate_gifs.utils import (
    REPO_ROOT, render_frame, save_gif
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_PATH = REPO_ROOT / "plots" / "Abrupt.gif"

TITLE1 = "Detect Abrupt Trends"
TITLE2 = "Detect Abrupt Trends with Padding"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate():
    df = pt.load_data("series_synthetic")[["date", "abrupt"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    value_col = "abrupt"

    print("Running detection (no padding) ...")
    res1 = pt.detect_trends(df.reset_index(), date_col="date", value_col=value_col, plot=False)
    segs1 = res1.segments

    print("Running detection (padding=28) ...")
    res2 = pt.detect_trends(
        df.reset_index(), date_col="date", value_col=value_col,
        plot=False, method_params=dict(abrupt_padding=28),
    )
    segs2 = res2.segments

    frames: list[Image.Image] = []
    durations: list[int] = []

    def R(title, sweep=None, segs=None, ranks=False, ra=1.0, sa=0.4):
        frames.append(render_frame(df, value_col, title, sweep, segs, ranks, ra, 12, sa))

    def hold(ms):
        durations.append(ms)

    # ── Cycle 1: no padding ────────────────────────────────────────────
    print("Rendering Cycle 1 ...")

    # 1. Raw plot (white background)
    R(TITLE1); hold(500)

    # 2. All segments sweep left to right (blue, green, blue, red, blue)
    for i in range(30):
        R(TITLE1, sweep=(i + 1) / 30, segs=segs1); hold(40)

    # 3. Sweep complete hold (all segments visible)
    R(TITLE1, sweep=1.0, segs=segs1); hold(500)

    # 4. Ranks fade in (larger, near top)
    for i in range(10):
        a = (i + 1) / 10
        R(TITLE1, sweep=1.0, segs=segs1, ranks=True, ra=a); hold(40)

    # 5. Result hold
    R(TITLE1, sweep=1.0, segs=segs1, ranks=True); hold(5000)

    # 6. Ranks fade out
    for i in range(10):
        a = max(0.0, 1.0 - (i + 1) / 10)
        R(TITLE1, sweep=1.0, segs=segs1, ranks=True, ra=a); hold(40)

    # ── Cycle 2: with padding ──────────────────────────────────────────
    print("Rendering Cycle 2 ...")

    # 7. Raw plot (new title)
    R(TITLE2); hold(500)

    # 8. All segments sweep left to right (padded)
    for i in range(30):
        R(TITLE2, sweep=(i + 1) / 30, segs=segs2); hold(40)

    # 9. Sweep complete hold
    R(TITLE2, sweep=1.0, segs=segs2); hold(500)

    # 10. Ranks fade in
    for i in range(10):
        a = (i + 1) / 10
        R(TITLE2, sweep=1.0, segs=segs2, ranks=True, ra=a); hold(40)

    # 11. Result hold
    R(TITLE2, sweep=1.0, segs=segs2, ranks=True); hold(5000)

    # 12. Ranks fade out
    for i in range(10):
        a = max(0.0, 1.0 - (i + 1) / 10)
        R(TITLE2, sweep=1.0, segs=segs2, ranks=True, ra=a); hold(40)

    # 13. Segments fade out (alpha fade, no sweep)
    for i in range(10):
        a = max(0.0, 1.0 - (i + 1) / 10)
        R(TITLE2, sweep=1.0, segs=segs2, sa=a * 0.4); hold(40)

    # 14. Brief pause on raw plot (matches frame 0 for seamless loop)
    R(TITLE1); hold(300)

    # ── Save ───────────────────────────────────────────────────────────
    n = len(frames)
    total_s = sum(durations) / 1000
    print(f"Saving {n} frames ({total_s:.1f}s total) ...")

    size_kb = save_gif(frames, durations, OUTPUT_PATH)
    print(f"Done -> {OUTPUT_PATH}  ({n} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    generate()
