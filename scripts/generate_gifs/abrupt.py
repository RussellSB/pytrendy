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
    REPO_ROOT, render_frame, save_gif, save_keyframes
)

def _crossfade(bottom: Image.Image, top: Image.Image, alpha: float) -> Image.Image:
    """Fade top image out to reveal bottom. alpha=0 shows top, alpha=1 shows bottom."""
    return Image.blend(top.convert("RGBA"), bottom.convert("RGBA"), alpha)

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

    def R1(title, sweep=None, segs=None, ranks=False, ra=1.0, sa=0.4):
        """Cycle 1 (no padding): ranks in 5-15 y-axis band."""
        frames.append(render_frame(df, value_col, title, sweep, segs, ranks, ra, 22, sa,
                                   rank_y_offset=0.876, rank_bold=False))

    def R2(title, sweep=None, segs=None, ranks=False, ra=1.0, sa=0.4):
        """Cycle 2 (with padding): ranks in 50-60 y-axis band."""
        frames.append(render_frame(df, value_col, title, sweep, segs, ranks, ra, 22, sa,
                                   rank_y_offset=0.367, rank_bold=False))

    def hold(ms):
        durations.append(ms)

    # ── Cycle 1: no padding ────────────────────────────────────────────
    print("Rendering Cycle 1 ...")

    # 1. Raw plot (white background)
    R1(TITLE1); hold(500)

    # 2. All segments sweep left to right (blue, green, blue, red, blue)
    for i in range(30):
        R1(TITLE1, sweep=(i + 1) / 30, segs=segs1); hold(40)

    # 3. Sweep complete hold (all segments visible)
    R1(TITLE1, sweep=1.0, segs=segs1); hold(500)

    # 4. Ranks fade in (larger, near top)
    for i in range(10):
        a = (i + 1) / 10
        R1(TITLE1, sweep=1.0, segs=segs1, ranks=True, ra=a); hold(40)

    # 5. Result hold
    R1(TITLE1, sweep=1.0, segs=segs1, ranks=True); hold(5000)

    # 6. Ranks fade out
    for i in range(10):
        a = max(0.0, 1.0 - (i + 1) / 10)
        R1(TITLE1, sweep=1.0, segs=segs1, ranks=True, ra=a); hold(40)

    # ── Crossfade: Phase 1 end → Phase 2 start ─────────────────────────
    # Pre-render the two frames to crossfade between
    phase1_end = render_frame(df, value_col, TITLE1, sweep_progress=1.0, segments=segs1)
    phase2_start = render_frame(df, value_col, TITLE2)

    for i in range(15):
        alpha = (i + 1) / 15
        frames.append(_crossfade(phase2_start, phase1_end, alpha))
        hold(50)

    # ── Cycle 2: with padding ──────────────────────────────────────────
    print("Rendering Cycle 2 ...")

    # 7. Raw plot (new title)
    R2(TITLE2); hold(500)

    # 8. All segments sweep left to right (padded)
    for i in range(30):
        R2(TITLE2, sweep=(i + 1) / 30, segs=segs2); hold(40)

    # 9. Sweep complete hold
    R2(TITLE2, sweep=1.0, segs=segs2); hold(500)

    # 10. Ranks fade in
    for i in range(10):
        a = (i + 1) / 10
        R2(TITLE2, sweep=1.0, segs=segs2, ranks=True, ra=a); hold(40)

    # 11. Result hold
    R2(TITLE2, sweep=1.0, segs=segs2, ranks=True); hold(5000)

    # 12. Ranks fade out
    for i in range(10):
        a = max(0.0, 1.0 - (i + 1) / 10)
        R2(TITLE2, sweep=1.0, segs=segs2, ranks=True, ra=a); hold(40)

    # ── Crossfade: Phase 2 end → Phase 1 start (seamless loop) ────────
    phase2_end = render_frame(df, value_col, TITLE2, sweep_progress=1.0, segments=segs2)
    phase1_start = render_frame(df, value_col, TITLE1)

    for i in range(15):
        alpha = (i + 1) / 15
        frames.append(_crossfade(phase1_start, phase2_end, alpha))
        hold(50)

    # ── Save keyframes for review ─────────────────────────────────────
    cycle1_result = render_frame(df, value_col, TITLE1, sweep_progress=1.0, segments=segs1,
                                 show_ranks=True, rank_alpha=1.0, rank_size=22,
                                 rank_y_offset=0.876, rank_bold=False)
    cycle2_result = render_frame(df, value_col, TITLE2, sweep_progress=1.0, segments=segs2,
                                 show_ranks=True, rank_alpha=1.0, rank_size=22,
                                 rank_y_offset=0.367, rank_bold=False)
    save_keyframes({
        "cycle1_result": cycle1_result,
        "cycle1_end": phase1_end,
        "phase2_start": phase2_start,
        "cycle2_result": cycle2_result,
        "cycle2_end": phase2_end,
    }, "Abrupt")

    # ── Save ───────────────────────────────────────────────────────────
    n = len(frames)
    total_s = sum(durations) / 1000
    print(f"Saving {n} frames ({total_s:.1f}s total) ...")

    size_kb = save_gif(frames, durations, OUTPUT_PATH)
    print(f"Done -> {OUTPUT_PATH}  ({n} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    generate()
