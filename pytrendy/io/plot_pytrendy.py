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
    for i, seg in enumerate(segments_enhanced):
        start = pd.to_datetime(seg['start'])
        end = pd.to_datetime(seg['end'])
        color = color_map.get(seg['direction'], 'gray')

        if ('trend_class' in seg and seg['trend_class'] == 'abrupt') or seg['direction'] == 'Noise': 
            start = start # Conditional logic for making abrupt visually tighter
        else: start = start - pd.Timedelta(days=1) # Everything else displaced left start

        # Get context on next seg if possible
        next_seg = segments_enhanced[i+1] if i+1 < len(segments_enhanced) else None
        neighbouring = next_seg and (pd.to_datetime(next_seg['start']) == (end + pd.Timedelta(days=1)))

        # Adjust neighbouring segment before abrupt or noise (visually). Avoid white lines
        next_seg_abrupt = next_seg and ((('trend_class' in next_seg) and (next_seg['trend_class'] == 'abrupt')) or next_seg['direction'] == 'Noise')
        if next_seg_abrupt and neighbouring:
            end = end + pd.Timedelta(days=1)
        else: end = end

        mask = (df.index >= start) & (df.index <= end) 
        ax.fill_between(df.index[mask], ymin, ymax, color=color, alpha=0.4)
        
        # Add ranking if up/down trend
        if 'change_rank' in seg:
            mid_date = start + (end - start) / 2
            y_pos = ymax - (ymax - ymin) * 0.05
            ax.text(mid_date, y_pos, str(seg['change_rank']), fontsize=12,
                    fontweight='bold', ha='center', va='top',
                    color=color[5:])
            
        # Add vertical line if next seg is same & touching
        if next_seg and neighbouring and next_seg['direction'] == seg['direction']:
            line_date = pd.to_datetime(seg['end'])
            ax.axvline(x=line_date, color=color[5:], linewidth=0.5)

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
        mpatches.Patch(color='lightgray', alpha=0.4, label='Noise'), 
    ]
    ax.legend(handles=legend_handles, loc='upper right', 
            bbox_to_anchor=(1, 1.15), ncol=4, frameon=True)

    plt.tight_layout()
    plt.show()