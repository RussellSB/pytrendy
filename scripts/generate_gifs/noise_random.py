#!/usr/bin/env python3
"""Generate Noise-Random.gif programmatically.

Recreates the animated GIF showing PyTrendy's trend detection
in noisy data with random noise, in a single-cycle animation.

Usage:
    python scripts/generate_gifs/noise_random.py

Output:
    plots/Noise-Random.gif
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
OUTPUT_PATH = REPO_ROOT / "plots" / "Noise-Random.gif"

TITLE = "Detect Trends in Noisy Data"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate():
    df = pt.load_data("series_synthetic")[["date", "gradual-noisy-20"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    value_col = "gradual-noisy-20"

    print("Running detection ...")
    res = pt.detect_trends(df.reset_index(), date_col="date", value_col=value_col, plot=False)
    segs = res.segments

    frames: list[Image.Image] = []
    durations: list[int] = []

    def R(title, sweep=None, segs=None, ranks=False, ra=1.0, sa=0.4):
        frames.append(render_frame(df, value_col, title, sweep, segs, ranks, ra, 12, sa))

    def hold(ms):
        durations.append(ms)

    # ── Single cycle ───────────────────────────────────────────────────
    print("Rendering animation ...")

    # 1. Raw plot (white background)
    R(TITLE); hold(500)

    # 2. All segments sweep left to right (blue, green, blue, red, blue)
    for i in range(30):
        R(TITLE, sweep=(i + 1) / 30, segs=segs); hold(40)

    # 3. Sweep complete hold (all segments visible)
    R(TITLE, sweep=1.0, segs=segs); hold(500)

    # 4. Ranks fade in (larger, near top)
    for i in range(10):
        a = (i + 1) / 10
        R(TITLE, sweep=1.0, segs=segs, ranks=True, ra=a); hold(40)

    # 5. Result hold
    R(TITLE, sweep=1.0, segs=segs, ranks=True); hold(5000)

    # 6. Ranks fade out
    for i in range(10):
        a = max(0.0, 1.0 - (i + 1) / 10)
        R(TITLE, sweep=1.0, segs=segs, ranks=True, ra=a); hold(40)

    # 7. Segments fade out (alpha fade, no sweep)
    for i in range(10):
        a = max(0.0, 1.0 - (i + 1) / 10)
        R(TITLE, sweep=1.0, segs=segs, sa=a * 0.4); hold(40)

    # 8. Brief pause on raw plot (matches frame 0 for seamless loop)
    R(TITLE); hold(300)

    # ── Save ───────────────────────────────────────────────────────────
    n = len(frames)
    total_s = sum(durations) / 1000
    print(f"Saving {n} frames ({total_s:.1f}s total) ...")

    size_kb = save_gif(frames, durations, OUTPUT_PATH)
    print(f"Done -> {OUTPUT_PATH}  ({n} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    generate()
