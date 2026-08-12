#!/usr/bin/env python3
"""Generate Noise-Spikes.gif programmatically.

Recreates the animated GIF showing PyTrendy's trend detection
in data with spikes. Starts from the gradual baseline (no spikes)
and progressively introduces each spike chronologically, using
full-image crossfades between states.

Usage:
    python scripts/generate_gifs/noise_spikes.py

Output:
    plots/Noise-Spikes.gif
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
OUTPUT_PATH = REPO_ROOT / "plots" / "Noise-Spikes.gif"

TITLE = "Detect Spikes"

# Spike dates to introduce chronologically (from notebook)
SPIKE_DATES = ["2025-04-08", "2025-05-08", "2025-06-08"]
SPIKE_VALUE = 200


def _crossfade(bottom: Image.Image, top: Image.Image, alpha: float) -> Image.Image:
    """Fade top image out to reveal bottom. alpha=0 shows top, alpha=1 shows bottom."""
    return Image.blend(top.convert("RGBA"), bottom.convert("RGBA"), alpha)


def _make_spike_df(base, num_spikes):
    """Return a copy of the base df with the first `num_spikes` spikes applied."""
    df = base.copy()
    df["gradual_spiky"] = df["gradual"]
    for date in SPIKE_DATES[:num_spikes]:
        df.loc[date:date, "gradual_spiky"] = SPIKE_VALUE
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate():
    # Load gradual data (base signal)
    df_base = pt.load_data("series_synthetic")[["date", "gradual"]]
    df_base["date"] = pd.to_datetime(df_base["date"])
    df_base = df_base.set_index("date")

    value_col = "gradual_spiky"

    # Build spike states: 0, 1, 2, 3 spikes
    spike_states = list(range(len(SPIKE_DATES) + 1))

    print("Running detection for each spike state ...")
    state_data = {}
    for num in spike_states:
        df_state = _make_spike_df(df_base, num)
        res = pt.detect_trends(df_state.reset_index(), date_col="date",
                               value_col=value_col, plot=False)
        state_data[num] = {
            "df": df_state,
            "segs": res.segments
        }
        print(f"  {num} spike(s): {len(res.segments)} segments detected")

    # Pre-render the key frame for each spike state (signal + segments + ranks)
    print("Pre-rendering key frames ...")
    key_frames = {}
    for num in spike_states:
        data = state_data[num]
        key_frames[num] = render_frame(
            data["df"], value_col, TITLE,
            sweep_progress=1.0, segments=data["segs"],
            show_ranks=True, rank_alpha=1.0
        )
        print(f"  {num} spike(s): rendered")

    frames: list[Image.Image] = []
    durations: list[int] = []

    def hold(ms):
        durations.append(ms)

    # ── Animation ──────────────────────────────────────────────────────
    print("Rendering animation ...")

    crossfade_frames = 15  # frames for each crossfade
    hold_ms = 50  # ms per frame during crossfade
    result_hold_ms = 2000  # ms to hold each result

    for idx, num in enumerate(spike_states):
        print(f"  Phase {idx + 1}: {num} spike(s)")

        # Hold the result frame
        frames.append(key_frames[num])
        hold(result_hold_ms)

        # Crossfade to next spike state: current fades out, next is background
        if idx < len(spike_states) - 1:
            next_num = spike_states[idx + 1]

            for i in range(crossfade_frames):
                alpha = (i + 1) / crossfade_frames
                blended = _crossfade(key_frames[next_num], key_frames[num], alpha)
                frames.append(blended)
                hold(hold_ms)

    # Hold final state (all spikes) longer
    frames.append(key_frames[spike_states[-1]])
    hold(3000)

    # Fade from final state back to starting frame for seamless loop
    for i in range(15):
        alpha = (i + 1) / 15
        blended = _crossfade(key_frames[spike_states[0]], key_frames[spike_states[-1]], alpha)
        frames.append(blended)
        hold(hold_ms)

    # ── Save ───────────────────────────────────────────────────────────
    n = len(frames)
    total_s = sum(durations) / 1000
    print(f"Saving {n} frames ({total_s:.1f}s total) ...")

    size_kb = save_gif(frames, durations, OUTPUT_PATH)
    print(f"Done -> {OUTPUT_PATH}  ({n} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    generate()
