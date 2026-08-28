"""**Visualize Detected Trends Over Time Series**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches


def _safe_adjacent(index, pos, offset):
    """
    Safely get an adjacent index value with bounds checking.
    
    Args:
        index: The index to access
        pos: Current position in the index
        offset: Offset from current position (+1 or -1)
        
    Returns:
        The adjacent index value if within bounds, None otherwise
    """
    new_pos = pos + offset
    if 0 <= new_pos < len(index):
        return index[new_pos]
    return None

def plot_pytrendy(df: pd.DataFrame, value_col: str, segments_enhanced: list[dict], index_type: str = "date", suppress_show: bool = False, plot_params: dict = None) -> plt.Figure:
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
        index_type (str):
            The type of index passed by the user. Different index types require different logic. Currently Accepted Index Types are: "date", "integer", "float".
        suppress_show (bool, optional):
            If True, suppresses the automatic display of the plot with plt.show(). Defaults to False.
        plot_params (dict, optional):
            Optional dict to customise plot appearance. Supported keys:

            - **figsize** (`tuple`): Figure size as (width, height). Defaults to (20, 5).
            - **title** (`str`): Plot title. Defaults to "PyTrendy Detection".
            - **xlabel** (`str`): X-axis label. Defaults to "Date".
            - **ylabel** (`str`): Y-axis label. Defaults to "Value".
            - **colors** (`dict`): Dictionary mapping direction ('Up', 'Down', 'Flat', 'Noise') to matplotlib colors. Defaults to light variants.
            - **alpha** (`float`): Transparency level for shaded regions. Defaults to 0.4.
            - **grid** (`dict`): Grid configuration with keys 'visible' (bool), 'which' (str), 'color' (str), 'alpha' (float).
            - **legend_loc** (`str`): Legend location. Defaults to "upper right".
            - **legend_bbox_to_anchor** (`tuple`): Legend box anchor position. Defaults to (1, 1.15).

    Returns:
        matplotlib.figure.Figure:
            The figure object containing the plot. Can be displayed with `plt.show()` or saved.
    """
    
    # Default plotting params
    default_params = {
        'figsize': (20, 5),
        'title': "PyTrendy Detection",
        'xlabel': "Date",
        'ylabel': "Value",
        'colors': {
            'Up': 'lightgreen',
            'Down': 'lightcoral',
            'Flat': 'lightblue',
            'Noise': 'lightgray',
        },
        'alpha': 0.4,
        'grid': {'visible': True, 'which': 'major', 'color': 'gray', 'alpha': 0.3},
        'legend_loc': 'upper right',
        'legend_bbox_to_anchor': (1, 1.15)
    }
    if plot_params:
        plot_params = dict(plot_params)  # avoid mutating caller's dict
        has_custom_legend_loc = 'legend_loc' in plot_params
        has_custom_legend_anchor = 'legend_bbox_to_anchor' in plot_params
        custom_colors = plot_params.pop('colors', None)
        custom_grid = plot_params.pop('grid', None)
        default_params.update(plot_params)
        if custom_colors:
            default_params['colors'].update(custom_colors)
        if custom_grid:
            default_params['grid'].update(custom_grid)
        if has_custom_legend_loc and not has_custom_legend_anchor:
            default_params['legend_bbox_to_anchor'] = None

    # Define colors
    color_map = default_params['colors']

    fig, ax = plt.subplots(figsize=default_params['figsize'])

    # Plot the value line
    ax.plot(df.index, df[value_col], color='black', lw=1)


    # Add shaded regions with fill_between
    ymin, ymax = ax.get_ylim()  # get plot's visible y-range
    for i, seg in enumerate(segments_enhanced):
        
        if index_type == "date":
            start = pd.to_datetime(seg['start'])
            end = pd.to_datetime(seg['end'])
        else:
            start = seg['start']
            end = seg['end']
        
        color = color_map.get(seg['direction'], 'gray')

        # Get context on prev seg if possible
        prev_seg = segments_enhanced[i-1] if i-1 >= 0 else None
        if index_type == "date":
            prev_neighbouring = prev_seg and (pd.to_datetime(prev_seg['end']) == (start - pd.Timedelta(days=1)))
        elif index_type == 'string':
            prev_neighbouring = prev_seg and (prev_seg['end'] == df.index[df.index.get_loc(start) - 1])
        else:
            prev_neighbouring = prev_seg and (prev_seg['end'] == (start - 1))

        is_prev_not_trend = prev_seg and (not ('trend_class' in prev_seg))

        # Current seg context
        is_abrupt = ('trend_class' in seg and seg['trend_class'] == 'abrupt')
        is_noise = (seg['direction'] == 'Noise')
        is_not_trend = not ('trend_class' in seg)

        # Get context on next seg if possible
        next_seg = segments_enhanced[i+1] if i+1 < len(segments_enhanced) else None
        if index_type == 'date':
            next_neighbouring = next_seg and (pd.to_datetime(next_seg['start']) == (end + pd.Timedelta(days=1)))
        elif index_type == 'string':
            end_pos = df.index.get_loc(end)
            next_neighbouring = next_seg and (next_seg['start'] == _safe_adjacent(df.index, end_pos, 1))
        else:
            next_neighbouring = next_seg and (next_seg['start'] == (end + 1))
        
        next_seg_abrupt = next_seg and (('trend_class' in next_seg) and (next_seg['trend_class'] == 'abrupt'))
        next_seg_noise = next_seg and (next_seg['direction'] == 'Noise')

        # Adjust starts when appropriate
        if is_abrupt or is_noise: 
            pass  # Keep start as-is for abrupt/noise segments
        else: 
            if index_type == 'date':
                new_start = start - pd.Timedelta(days=1) # Everything else displaced left start
            elif index_type == 'string':
                start_pos = df.index.get_loc(start)
                new_start = _safe_adjacent(df.index, start_pos, -1)
            else:
                new_start = start - 1 # Everything else displaced left start

            # Check validity of plot start adjustment
            value_new_start = df.loc[new_start, value_col] if new_start is not None and new_start in df.index else None

            value = df.loc[start, value_col]

            valid_up_start = (value_new_start) and (seg['direction'] == 'Up') and (value_new_start < value)
            valid_down_start = (value_new_start) and (seg['direction'] == 'Down') and (value_new_start > value)
            if valid_up_start or valid_down_start or is_not_trend:
                start = new_start # Apply left displacement only if valid
            else: 
                # if not displaced and prev is not trend, adjust by plotting (as prev has already been drawn)
                if is_prev_not_trend and prev_neighbouring:
                    if index_type == 'date':
                        prev_end = pd.to_datetime(segments_enhanced[i-1]['end'])
                        prev_new_end = (prev_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    elif index_type == 'string':
                        prev_end = segments_enhanced[i-1]['end']
                        prev_end_pos = df.index.get_loc(prev_end)
                        prev_new_end = _safe_adjacent(df.index, prev_end_pos, 1)
                    else:
                        prev_end = segments_enhanced[i-1]['end']
                        prev_new_end = prev_end + 1
                    
                    if prev_new_end is not None:
                        if index_type == 'string':
                            mask = (np.arange(len(df)) >= df.index.get_loc(prev_end)) & (np.arange(len(df)) <= df.index.get_loc(prev_new_end))
                        else:
                            mask = (df.index >= prev_end) & (df.index <= prev_new_end)
                        prev_color = color_map.get(segments_enhanced[i-1]['direction'], 'gray')
                        ax.fill_between(df.index[mask], ymin, ymax, color=prev_color, alpha=0.4)

        # Adjust ends when appropriate
        if (next_seg_abrupt or next_seg_noise) and next_neighbouring:
            if index_type == 'date':
                new_end = end + pd.Timedelta(days=1)
            elif index_type == 'string':
                end_pos = df.index.get_loc(end)
                new_end = _safe_adjacent(df.index, end_pos, 1)
            else:
                new_end = end + 1
            
            # Check validity of plot end adjustment
            value_new_end = df.loc[new_end, value_col] if new_end is not None and new_end in df.index else None
            value = df.loc[end, value_col]

            valid_up_end = (value_new_end) and (seg['direction'] == 'Up') and (value_new_end > value)
            valid_down_end = (value_new_end) and (seg['direction'] == 'Down') and (value_new_end < value)
            is_not_trend = not ('trend_class' in seg)
            if valid_up_end or valid_down_end or is_not_trend:
                end = new_end  # Apply right displacement only if valid
            else: 
                # if not displaced and next is noise, adjust for next plotting round
                if next_seg_noise and next_neighbouring: 
                    if index_type == 'date':
                        segments_enhanced[i+1]['start'] = (pd.to_datetime(segments_enhanced[i+1]['start']) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    elif index_type == 'string':
                        next_start_pos = df.index.get_loc(segments_enhanced[i+1]['start'])
                        segments_enhanced[i+1]['start'] = _safe_adjacent(df.index, next_start_pos, -1)
                    else:
                        segments_enhanced[i+1]['start'] = (segments_enhanced[i+1]['start'] - 1)
        else: 
            pass  # Keep end as-is

        if index_type == 'string':
            mask = (np.arange(len(df)) >= df.index.get_loc(start)) & (np.arange(len(df)) <= df.index.get_loc(end))
        else:
            mask = (df.index >= start) & (df.index <= end) 


        ax.fill_between(df.index[mask], ymin, ymax, color=color, alpha=0.4)
        
        # Add ranking if up/down trend
        if 'change_rank' in seg and seg['direction'] in ['Up', 'Down']:
            
            if index_type in ['string']:
                midpoint = int((df.index.get_loc(end) - df.index.get_loc(start))/2)
                mid_date = df.index[df.index.get_loc(start) + midpoint]
            else:
                mid_date = start + (end - start) / 2

            
            y_pos = ymax - (ymax - ymin) * 0.05
            ax.text(mid_date, y_pos, str(seg['change_rank']), fontsize=12,
                    fontweight='bold', ha='center', va='top',
                    color=color[5:])
            
        # Add vertical line if next seg is same & touching
        if next_seg and next_neighbouring and next_seg['direction'] == seg['direction']:
            if index_type == 'date':
                line_date = pd.to_datetime(seg['end'])
            else:
                line_date = seg['end']
            ax.axvline(x=line_date, color=color[5:], linewidth=0.5)

    # Set limits
    if index_type == 'string':
        first_date = df.index[0]
        last_date = df.index[-1]
    else:
        first_date = df.index.min()
        last_date = df.index.max()

    ax.set_xlim(first_date, last_date)
    ax.set_ylim(ymin, ymax)

    if index_type == 'date':
        # Major ticks: every 7 days (with labels)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Minor ticks: every day (no labels, just tick marks/grid)
        ax.xaxis.set_minor_locator(mdates.DayLocator())

    # Rotate major tick labels
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

    # Optional: show grid lines for both
    grid_cfg = default_params['grid']
    if grid_cfg.get('visible', True):
        ax.grid(True, which=grid_cfg.get('which', 'major'),
                color=grid_cfg.get('color', 'gray'), alpha=grid_cfg.get('alpha', 0.3))
    else:
        ax.grid(False)

    if index_type == 'string':
        ticks = ax.get_xticks()
        labels = [t.get_text() for t in ax.get_xticklabels()]
        n = 10
        ax.set_xticks(ticks[::n])
        ax.set_xticklabels(labels[::n], rotation=90, ha='center')


    ax.set_title(default_params['title'], fontsize=20)

    if index_type == 'date':
        ax.set_xlabel(default_params.get('xlabel', 'Date'))
    elif index_type == 'string':
        ax.set_xlabel(default_params.get('xlabel', 'Label'))
    else:
        ax.set_xlabel(default_params.get('xlabel', 'Index'))

    ax.set_ylabel(default_params.get('ylabel', 'Value'))

    # Create custom legend handles (colored boxes)
    legend_handles = [
        mpatches.Patch(color=default_params['colors']['Up'], alpha=default_params['alpha'], label='Up'),
        mpatches.Patch(color=default_params['colors']['Down'], alpha=default_params['alpha'], label='Down'),
        mpatches.Patch(color=default_params['colors']['Flat'], alpha=default_params['alpha'], label='Flat'),
        mpatches.Patch(color=default_params['colors']['Noise'], alpha=default_params['alpha'], label='Noise'), 
    ]
    ax.legend(handles=legend_handles, loc=default_params['legend_loc'], 
            bbox_to_anchor=default_params['legend_bbox_to_anchor'], ncol=4, frameon=True)

    plt.tight_layout()
    if not suppress_show:
        plt.show()
    return fig