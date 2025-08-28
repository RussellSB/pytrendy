import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

def plot_pytrendy(df:pd.DataFrame, value_col: str, segments_enhanced:list):
    """Plot visuals of trend detected segments over signal of interest."""
    # Define colors
    color_map = {
        'Up': 'lightgreen',
        'Down': 'lightcoral',
        'Flat': 'lightblue',
        'Noise': 'lightgray',
    }

    fig, ax = plt.subplots(figsize=(20, 5))

    # Plot the value line
    ax.plot(df.index, df[value_col], color='black', lw=1)

    # Add shaded regions with fill_between
    ymin, ymax = ax.get_ylim()  # get plot's visible y-range
    for rank, seg in enumerate(segments_enhanced, start=1):
        start = pd.to_datetime(seg['start'])
        end = pd.to_datetime(seg['end'])
        color = color_map.get(seg['direction'], 'gray')

        mask = (df.index >= start - pd.Timedelta(days=1)) & (df.index <= end) # TODO: make work by pixels somehow
        ax.fill_between(df.index[mask], ymin, ymax, color=color, alpha=0.4)
        
        # Add ranking if up/down trend
        if seg['direction'] in ['Up', 'Down']:
            mid_date = start + (end - start) / 2
            y_pos = ymax - (ymax - ymin) * 0.05
            ax.text(mid_date, y_pos, str(rank), fontsize=12,
                    fontweight='bold', ha='center', va='top',
                    color=color[5:])

    # Set limits
    first_date = df.index.min()
    last_date = df.index.max()
    ax.set_xlim(first_date, last_date)
    ax.set_ylim(ymin, ymax)

    # Major ticks: every 7 days (with labels)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Minor ticks: every day (no labels, just tick marks/grid)
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    # Rotate major tick labels
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    # Optional: show grid lines for both
    ax.grid(True, which='major', color='gray', alpha=0.3)

    ax.set_title("PyTrendy Detection", fontsize=20)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")

    # Create custom legend handles (colored boxes)
    legend_handles = [
        mpatches.Patch(color='lightgreen', alpha=0.4, label='Up'),
        mpatches.Patch(color='lightcoral', alpha=0.4, label='Down'),
        mpatches.Patch(color='lightblue', alpha=0.4, label='Flat'),
        mpatches.Patch(color='lightgray', alpha=0.4, label='Noise'), # TODO: Show optionally later, based on up_only, down_only, robustness=False ... etc
    ]
    ax.legend(handles=legend_handles, loc='upper right', 
            bbox_to_anchor=(1, 1.15), ncol=4, frameon=True)

    plt.tight_layout()
    plt.show()