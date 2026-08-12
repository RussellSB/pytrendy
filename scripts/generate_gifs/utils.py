#!/usr/bin/env python3
"""Shared utilities for GIF generation scripts.

Contains common functions and constants used by both abrupt.py and gradual.py
for generating programmatic GIFs showing PyTrendy's trend detection.

Boundary adjustment logic copied from plot_pytrendy (pytrendy/io/plot_pytrendy.py:50-111).
If plot_pytrendy ever updates this logic, this code would need to too.
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


# ---------------------------------------------------------------------------
# Boundary adjustment logic
# ---------------------------------------------------------------------------
def adjust_segment_boundaries(df, value_col, segments):
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
            else:
                # Fallback: if not displaced and prev is not trend, extend prev end to cover gap
                # Copied from plot_pytrendy (pytrendy/io/plot_pytrendy.py:80-87)
                if is_prev_not_trend and prev_neighbouring:
                    prev_idx = i - 1
                    prev_adj_end = adjusted[prev_idx][1]
                    adjusted[prev_idx] = (adjusted[prev_idx][0], prev_adj_end + pd.Timedelta(days=1))

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
                # Fallback: if not displaced and next is noise, shift next start left to close gap
                # Copied from plot_pytrendy (pytrendy/io/plot_pytrendy.py:104-106)
                if next_seg_noise and next_neighbouring:
                    next_seg["start"] = (pd.to_datetime(next_seg["start"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            end = end

        adjusted.append((start, end))

    return adjusted


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_frame(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    sweep_progress: float | None = None,
    segments: list[dict] | None = None,
    show_ranks: bool = False,
    rank_alpha: float = 1.0,
    rank_size: int = 12,
    seg_alpha: float = 0.4,
    rank_y_offset: float = 0.05,
    rank_bold: bool = True,
    rank_center_on_data: bool = False,
    title_suffix: str | None = None,
) -> Image.Image:
    """Render one frame as a complete matplotlib figure -> PIL Image.

    Args:
        sweep_progress: If set (0-1), clip ALL segments to this fraction
                        of the x-range (left-to-right chronological sweep).
        segments: Segment dicts to render as filled regions.
        show_ranks: Whether to show change_rank numbers.
        rank_alpha: Opacity of rank numbers (0-1).
        rank_size: Font size for rank numbers.
        seg_alpha: Opacity of segment fills (0-1).
        rank_y_offset: Rank vertical position as fraction from top of y-range
                        (e.g. 0.05 = 5% from top, 0.95 = near bottom).
        rank_bold: Whether rank numbers use bold font weight.
        rank_center_on_data: If True, vertically centre each rank within its
                        segment's data value range instead of a fixed plot offset.
        title_suffix: Optional suffix text drawn right of the title in light gray
                        (e.g. "(seed=10, std=20)").
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
        adjusted = adjust_segment_boundaries(df, value_col, segments)
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
                if rank_center_on_data:
                    seg_mask = (df.index >= start) & (df.index <= end)
                    vals = df.loc[seg_mask, value_col]
                    seg_min, seg_max = vals.min(), vals.max()
                    if seg_max > seg_min:
                        y_pos = seg_max - (seg_max - seg_min) * 0.5
                    else:
                        y_pos = ymax - (ymax - ymin) * rank_y_offset
                else:
                    y_pos = ymax - (ymax - ymin) * rank_y_offset
                rc = RANK_COLORS.get(seg["direction"], (0, 0, 0))
                rc_norm = (rc[0]/255, rc[1]/255, rc[2]/255)
                ax.text(mid, y_pos, str(seg["change_rank"]), fontsize=rank_size,
                        fontweight="bold" if rank_bold else "normal",
                        ha="center", va="center",
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
    if title_suffix:
        fig.canvas.draw()  # ensure renderer available for extent calc
        renderer = fig.canvas.get_renderer()
        bb = ax.title.get_window_extent(renderer=renderer)
        fig.text(
            bb.x1 + 8, bb.y0 + bb.height / 2,
            title_suffix, fontsize=15, color="lightgray",
            ha="left", va="center",
        )
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


def save_gif(frames, durations, output_path):
    """Save frames as an optimised GIF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    palette_frames = [
        im.convert("RGB").quantize(colors=256, method=Image.Quantize.FASTOCTREE)
        for im in frames
    ]

    palette_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=palette_frames[1:],
        loop=0,
        duration=durations,
        optimize=True,
    )

    size_kb = output_path.stat().st_size / 1024
    return size_kb
