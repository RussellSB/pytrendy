#!/usr/bin/env python3
"""Generate Noise-Random.gif programmatically.

Recreates the animated GIF showing PyTrendy's trend detection
in noisy data with random noise, showing how detection adapts
as noise levels increase.

Usage:
    python scripts/generate_gifs/noise_random.py

Output:
    plots/Noise-Random.gif
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytrendy as pt
from scripts.generate_gifs.utils import (
    REPO_ROOT, render_frame, save_gif, save_keyframes
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_PATH = REPO_ROOT / "plots" / "Noise-Random.gif"

TITLE = "Detect Noise"

# Noise levels to cycle through (from notebook)
NOISE_LEVELS = [0, 10, 20, 50]
SEED = 10


def _crossfade(bottom: Image.Image, top: Image.Image, alpha: float) -> Image.Image:
    """Fade top image out to reveal bottom. alpha=0 shows top, alpha=1 shows bottom."""
    return Image.blend(top.convert("RGBA"), bottom.convert("RGBA"), alpha)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate():
    # Load gradual data (base signal)
    df_base = pt.load_data("series_synthetic")[["date", "gradual"]]
    df_base["date"] = pd.to_datetime(df_base["date"])
    df_base = df_base.set_index("date")

    # Pre-generate noise for consistent random seed
    rng = np.random.default_rng(seed=SEED)

    # Run detection for each noise level
    print("Running detection for each noise level ...")
    noise_data = {}
    for noise_std in NOISE_LEVELS:
        df_noisy = df_base.copy()
        if noise_std > 0:
            df_noisy["gradual"] = df_noisy["gradual"] + rng.normal(0, noise_std, size=len(df_noisy))
        res = pt.detect_trends(df_noisy.reset_index(), date_col="date", value_col="gradual", plot=False)
        noise_data[noise_std] = {
            "df": df_noisy,
            "segs": res.segments
        }
        print(f"  Noise std={noise_std}: {len(res.segments)} segments detected")

    # Pre-render the key frame for each noise level (signal + segments + ranks)
    print("Pre-rendering key frames ...")
    key_frames = {}
    for noise_std in NOISE_LEVELS:
        data = noise_data[noise_std]
        suffix = f"(seed={SEED}, std={noise_std})"
        key_frames[noise_std] = render_frame(
            data["df"], "gradual", TITLE,
            sweep_progress=1.0, segments=data["segs"],
            show_ranks=True, rank_alpha=1.0,
            title_suffix=suffix
        )
        print(f"  Noise std={noise_std}: rendered")

    save_keyframes(key_frames, "Noise-Random")

    frames: list[Image.Image] = []
    durations: list[int] = []

    def hold(ms):
        durations.append(ms)

    # ── Animation ──────────────────────────────────────────────────────
    print("Rendering animation ...")

    crossfade_frames = 15  # frames for each crossfade
    hold_ms = 50  # ms per frame during crossfade
    result_hold_ms = 2000  # ms to hold each result

    for idx, noise_std in enumerate(NOISE_LEVELS):
        print(f"  Phase {idx + 1}: Noise std={noise_std}")

        # Hold the result frame
        frames.append(key_frames[noise_std])
        hold(result_hold_ms)

        # Crossfade to next noise level: current fades out, next is background
        if idx < len(NOISE_LEVELS) - 1:
            next_std = NOISE_LEVELS[idx + 1]

            for i in range(crossfade_frames):
                alpha = (i + 1) / crossfade_frames
                # next image is background, current image fades out on top
                blended = _crossfade(key_frames[next_std], key_frames[noise_std], alpha)
                frames.append(blended)
                hold(hold_ms)

    # Hold final state (high noise) longer
    frames.append(key_frames[NOISE_LEVELS[-1]])
    hold(3000)

    # Fade from final state back to starting frame for seamless loop
    for i in range(15):
        alpha = (i + 1) / 15
        blended = _crossfade(key_frames[NOISE_LEVELS[0]], key_frames[NOISE_LEVELS[-1]], alpha)
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
