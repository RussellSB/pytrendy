#!/usr/bin/env python3
"""Generate Abrupt.gif programmatically.

Recreates the animated GIF showing PyTrendy's abrupt trend detection,
with and without padding, in a two-cycle animation.

Usage:
    python scripts/generate_gifs/abrupt.py

Output:
    plots/Abrupt.gif
"""

import io
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytrendy as pt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "plots" / "Abrupt.gif"

TARGET_W, TARGET_H = 1906, 600
FIGSIZE = (20, 5.6)
DPI = 100
TOP_PAD = 7  # pixels of padding above the plot

COLOR_MAP = {
    "Up": "lightgreen",
    "Down": "lightcoral",
    "Flat": "lightblue",
    "Noise": "lightgray",
}

# Darker colours for rank text (won't get corrupted by palette quantisation)
RANK_COLORS = {
    "Up": (0, 100, 0),      # dark green
    "Down": (180, 0, 0),    # dark red
}

TITLE1 = "Detect Abrupt Trends"
TITLE2 = "Detect Abrupt Trends with Padding"


# ---------------------------------------------------------------------------
# Boundary adjustment logic
# ---------------------------------------------------------------------------
def _adjust_segment_boundaries(df, value_col, segments):
    """Return list of (start, end) tuples with boundary adjustments applied.

    Boundary adjustment logic copied from plot_pytrendy (pytrendy/io/plot_pytrendy.py:50-111).
    If plot_pytrendy ever updates this logic, this code would need to too.
    """
    adjusted = []
    for i, seg in enumerate(segments):
        start = pd.to_datetime(seg["start"])
        end = pd.to_datetime(seg["end"])

        # Get context on prev seg if possible
        prev_seg = segments[i-1] if i-1 >= 0 else None
        prev_neighbouring = prev_seg and (pd.to_datetime(prev_seg["end"]) == (start - pd.Timedelta(days=1)))
        is_prev_not_trend = prev_seg and (not ("trend_class" in prev_seg))

        # Current seg context
        is_abrupt = ("trend_class" in seg and seg["trend_class"] == "abrupt")
        is_noise = (seg["direction"] == "Noise")
        is_not_trend = not ("trend_class" in seg)

        # Get context on next seg if possible
        next_seg = segments[i+1] if i+1 < len(segments) else None
        next_neighbouring = next_seg and (pd.to_datetime(next_seg["start"]) == (end + pd.Timedelta(days=1)))
        next_seg_abrupt = next_seg and (("trend_class" in next_seg) and (next_seg["trend_class"] == "abrupt"))
        next_seg_noise = next_seg and (next_seg["direction"] == "Noise")

        # Adjust starts when appropriate
        if is_abrupt or is_noise:
            start = start  # Conditional logic for making abrupt visually tighter
        else:
            new_start = start - pd.Timedelta(days=1)  # Everything else displaced left start

            # Check validity of plot start adjustment
            value_new_start = df.loc[new_start, value_col] if new_start in df.index else None
            value = df.loc[start, value_col]

            valid_up_start = (value_new_start) and (seg["direction"] == "Up") and (value_new_start < value)
            valid_down_start = (value_new_start) and (seg["direction"] == "Down") and (value_new_start > value)
            if valid_up_start or valid_down_start or is_not_trend:
                start = new_start  # Apply left displacement only if valid

        # Adjust ends when appropriate
        if (next_seg_abrupt or next_seg_noise) and next_neighbouring:
            new_end = end + pd.Timedelta(days=1)

            # Check validity of plot end adjustment
            value_new_end = df.loc[new_end, value_col] if new_end in df.index else None
            value = df.loc[end, value_col]

            valid_up_end = (value_new_end) and (seg["direction"] == "Up") and (value_new_end > value)
            valid_down_end = (value_new_end) and (seg["direction"] == "Down") and (value_new_end < value)
            is_not_trend = not ("trend_class" in seg)
            if valid_up_end or valid_down_end or is_not_trend:
                end = new_end  # Apply right displacement only if valid
        else:
            end = end

        adjusted.append((start, end))

    return adjusted


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_frame(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    sweep_progress: float | None = None,
    segments: list[dict] | None = None,
    show_ranks: bool = False,
    rank_alpha: float = 1.0,
    rank_size: int = 12,
    seg_alpha: float = 0.4,
) -> Image.Image:
    """Render one frame as a complete matplotlib figure -> PIL Image.

    Args:
        sweep_progress: If set (0-1), clip ALL segments to this fraction
                        of the x-range (left-to-right chronological sweep).
        segments: Segment dicts to render as filled regions.
        show_ranks: Whether to show change_rank numbers.
        rank_alpha: Opacity of rank numbers (0-1).
        rank_size: Font size for rank numbers.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.plot(df.index, df[value_col], color="black", lw=1)
    ymin, ymax = ax.get_ylim()

    first_date = pd.Timestamp(df.index.min())
    last_date = pd.Timestamp(df.index.max())
    total_span = last_date - first_date

    # Determine clip boundary for sweep
    if sweep_progress is not None and sweep_progress > 0:
        clip_end = first_date + total_span * min(sweep_progress, 1.0)
    else:
        clip_end = last_date  # no clipping

    # Render segments clipped to sweep range
    if segments:
        adjusted = _adjust_segment_boundaries(df, value_col, segments)
        for idx, seg in enumerate(segments):
            start, end = adjusted[idx]
            color = COLOR_MAP.get(seg["direction"], "gray")

            # Clip segment to sweep range
            vis_start = max(start, first_date)
            vis_end = min(end, clip_end)
            if vis_start < vis_end:
                mask = (df.index >= vis_start) & (df.index <= vis_end)
                ax.fill_between(df.index[mask], ymin, ymax, color=color, alpha=seg_alpha)

            # Render rank if sweep has passed this segment
            if (show_ranks and seg["direction"] in ("Up", "Down")
                    and "change_rank" in seg and sweep_progress is not None
                    and clip_end >= end):
                mid = start + (end - start) / 2
                y_pos = ymax - (ymax - ymin) * 0.05
                rc = RANK_COLORS.get(seg["direction"], (0, 0, 0))
                rc_norm = (rc[0]/255, rc[1]/255, rc[2]/255)
                ax.text(mid, y_pos, str(seg["change_rank"]), fontsize=rank_size,
                        fontweight="bold", ha="center", va="center",
                        color=rc_norm, alpha=rank_alpha)

    # Formatting
    ax.set_xlim(first_date, last_date)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
    ax.grid(True, which="major", color="gray", alpha=0.3)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    legend_handles = [
        mpatches.Patch(color="lightgreen", alpha=0.4, label="Up"),
        mpatches.Patch(color="lightcoral", alpha=0.4, label="Down"),
        mpatches.Patch(color="lightblue", alpha=0.4, label="Flat"),
        mpatches.Patch(color="lightgray", alpha=0.4, label="Noise"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1, 1.10), ncol=4, frameon=True)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    buf.close()
    plt.close(fig)

    # Resize plot to fit within target minus top padding
    plot_h = TARGET_H - TOP_PAD
    img = img.resize((TARGET_W, plot_h), Image.Resampling.LANCZOS)

    # Create canvas with transparent padding at top
    canvas = Image.new("RGBA", (TARGET_W, TARGET_H), (255, 255, 255, 0))
    canvas.paste(img, (0, TOP_PAD))
    return canvas


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
        frames.append(_render_frame(df, value_col, title, sweep, segs, ranks, ra, 12, sa))

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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    palette_frames = [
        im.convert("RGB").quantize(colors=256, method=Image.Quantize.FASTOCTREE)
        for im in frames
    ]

    palette_frames[0].save(
        str(OUTPUT_PATH),
        save_all=True,
        append_images=palette_frames[1:],
        loop=0,
        duration=durations,
        optimize=True,
    )

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Done -> {OUTPUT_PATH}  ({n} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    generate()
