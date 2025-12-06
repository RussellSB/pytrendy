"""**Visualize Detected Trends Over Time Series**"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

def plot_pytrendy(df: pd.DataFrame, value_col: str, segments_enhanced: list[dict], suppress_show: bool = False) -> plt.Figure:
    """
    Visualizes detected trend segments over the original time series signal.
    
    This function overlays shaded regions on the signal to indicate trends such as Up, Down, Flat, and Noise
    It also annotates ranked segments and handles visual adjustments for abrupt transitions.

    Args:
        df (pd.DataFrame):
            Time series data with datetime index and signal column.
        value_col (str):
            Name of the column containing the signal to plot.
        segments_enhanced (list):
            List of segment dictionaries containing keys like `'start'`, `'end'`, `'direction'`, `'trend_class'`, and `'change_rank'`.
        suppress_show (bool, optional):
            If True, suppresses the automatic display of the plot with plt.show(). Defaults to False.

    Returns:
        matplotlib.figure.Figure:
            The figure object containing the plot. Can be displayed with `plt.show()` or saved.
    """
    
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

        # Get context on prev seg if possible
        prev_seg = segments_enhanced[i-1] if i-1 >= 0 else None
        prev_neighbouring = prev_seg and (pd.to_datetime(prev_seg['end']) == (start - pd.Timedelta(days=1)))
        is_prev_not_trend = prev_seg and (not ('trend_class' in prev_seg))

        # Current seg context
        is_abrupt = ('trend_class' in seg and seg['trend_class'] == 'abrupt')
        is_noise = (seg['direction'] == 'Noise')
        is_not_trend = not ('trend_class' in seg)

        # Get context on next seg if possible
        next_seg = segments_enhanced[i+1] if i+1 < len(segments_enhanced) else None
        next_neighbouring = next_seg and (pd.to_datetime(next_seg['start']) == (end + pd.Timedelta(days=1)))
        next_seg_abrupt = next_seg and (('trend_class' in next_seg) and (next_seg['trend_class'] == 'abrupt'))
        next_seg_noise = next_seg and (next_seg['direction'] == 'Noise')

        # Adjust starts when appropriate
        if is_abrupt or is_noise: 
            start = start # Conditional logic for making abrupt visually tighter
        else: 
            new_start = start - pd.Timedelta(days=1) # Everything else displaced left start

            # Check validity of plot start adjustment
            value_new_start = df.loc[new_start, value_col] if new_start in df.index else None
            value = df.loc[start, value_col]

            valid_up_start = (value_new_start) and (seg['direction'] == 'Up') and (value_new_start < value)
            valid_down_start = (value_new_start) and (seg['direction'] == 'Down') and (value_new_start > value)
            if valid_up_start or valid_down_start or is_not_trend:
                start = new_start # Apply left displacement only if valid
            else: 
                # if not displaced and prev is not trend, adjust by plotting (as prev has already been drawn)
                if is_prev_not_trend and prev_neighbouring: 
                    prev_end = pd.to_datetime(segments_enhanced[i-1]['end'])
                    prev_new_end = (prev_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    mask = (df.index >= prev_end) & (df.index <= prev_new_end) 
                    prev_color = color_map.get(segments_enhanced[i-1]['direction'], 'gray')
                    ax.fill_between(df.index[mask], ymin, ymax, color=prev_color, alpha=0.4)


        # Adjust ends when appropriate
        if (next_seg_abrupt or next_seg_noise) and next_neighbouring:
            new_end = end + pd.Timedelta(days=1)
            
            # Check validity of plot end adjustment
            value_new_end = df.loc[new_end, value_col] if new_end in df.index else None
            value = df.loc[end, value_col]

            valid_up_end = (value_new_end) and (seg['direction'] == 'Up') and (value_new_end > value)
            valid_down_end = (value_new_end) and (seg['direction'] == 'Down') and (value_new_end < value)
            is_not_trend = not ('trend_class' in seg)
            if valid_up_end or valid_down_end or is_not_trend:
                end = new_end  # Apply right displacement only if valid
            else: 
                # if not displaced and next is noise, adjust for next plotting round
                if next_seg_noise and next_neighbouring: 
                    segments_enhanced[i+1]['start'] = (pd.to_datetime(segments_enhanced[i+1]['start']) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else: 
            end = end

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
        if next_seg and next_neighbouring and next_seg['direction'] == seg['direction']:
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
    if not suppress_show:
        plt.show()
    return fig