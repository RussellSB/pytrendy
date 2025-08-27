# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

def plot_pytrendy(df: pd.DataFrame, value_col: str, segments_enhanced: list):
    """Plot visuals of trend detected segments over signal of interest."""

    # Plot constants
    PAD_PX = 4.5 # how many screen pixels to extend on each side (tune 1..6)
    color_map = { # Colours per segment direction
        'Up': 'lightgreen',
        'Down': 'lightcoral',
        'Flat': 'lightblue',
        'Noise': 'lightgray',
    }

    fig, ax = plt.subplots(figsize=(20, 5))

    # Ensure axis limits are set so transforms are accurate:
    ax.set_xlim(df.index.min(), df.index.max())

    # Force a draw so transforms are initialized (required to convert pixel<->data).
    fig.canvas.draw()

    # blended transform: x in data coords (dates), y in axes coords (0..1)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    # Draw shaded bands (expanded by PAD_PX)
    for rank, seg in enumerate(segments_enhanced, start=1):
        start = pd.to_datetime(seg['start'])
        end   = pd.to_datetime(seg['end'])
        color = color_map.get(seg['direction'], 'gray')

        # convert date -> matplotlib float
        start_num = mdates.date2num(start)
        end_num   = mdates.date2num(end)

        # convert data->pixels (x axis); y value can be 0
        x0_px = ax.transData.transform((start_num, 0))[0]
        x1_px = ax.transData.transform((end_num,   0))[0]

        # expand by PAD_PX pixels on both sides
        x0_padded_px = x0_px - PAD_PX
        x1_padded_px = x1_px + PAD_PX

        # convert padded pixels back to data coords (date-number)
        x0_padded_num = ax.transData.inverted().transform((x0_padded_px, 0))[0]
        x1_padded_num = ax.transData.inverted().transform((x1_padded_px, 0))[0]

        # convert back to datetime
        start_padded = mdates.num2date(x0_padded_num)
        end_padded   = mdates.num2date(x1_padded_num)

        # draw full-height vertical band (no hairline edges)
        ax.axvspan(start_padded, end_padded, ymin=0, ymax=1,
                   facecolor=color, edgecolor=color, linewidth=0,
                   antialiased=False, alpha=0.4, zorder=0)

        # ranking label (use axes y coords via transform)
        if seg['direction'] in ['Up', 'Down']:
            mid_date = start + (end - start) / 2
            ax.text(mid_date, 0.95, str(rank),
                    transform=trans, fontsize=12, fontweight='bold',
                    ha='center', va='top', color=color[5:])

    # Plot the data line on top
    ax.plot(df.index, df[value_col], color='black', lw=1, zorder=2)

    # Axis formatting (same as before)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    ax.grid(True, which='major', color='gray', alpha=0.3)

    ax.set_title("PyTrendy Detection", fontsize=20)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")

    legend_handles = [
        mpatches.Patch(color='lightgreen', alpha=0.4, label='Up'),
        mpatches.Patch(color='lightcoral', alpha=0.4, label='Down'),
        mpatches.Patch(color='lightblue',  alpha=0.4, label='Flat'),
        mpatches.Patch(color='lightgray',  alpha=0.4, label='Noise'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              bbox_to_anchor=(1, 1.15), ncol=4, frameon=True)

    plt.tight_layout()
    plt.show()

    