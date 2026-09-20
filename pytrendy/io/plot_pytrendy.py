"""**Visualize Detected Trends Over Time Series**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors


# Tick-density calibration. All private (no public API).
# _MAX_PINNED_TICKS is the label-density target: pinned majors above it are
# thinned to every ceil(n / cap)th observation, and multi-day calendar majors
# scale their interval to keep labels near it. Lower = sparser, higher = denser.
_MAX_PINNED_TICKS = 40
# _WEEKLY_MAX_SPAN_DAYS / _BIWEEKLY_MAX_SPAN_DAYS are daily-span thresholds in
# calendar days: weekly majors up to the first, biweekly to the second, monthly
# beyond. Raise them to keep a finer major unit for longer spans.
_WEEKLY_MAX_SPAN_DAYS = 200
_BIWEEKLY_MAX_SPAN_DAYS = 400
# _MAX_MINOR_TICKS is the minor-ruler legibility target; matplotlib's hard
# ceiling is MAXTICKS (1000). Lower = sparser ruler, higher = denser.
_MAX_MINOR_TICKS = 900
# _YEAR_MAJOR_MIN_SPAN_DAYS: monthly-or-coarser cadences switch from month to
# year majors once the span reaches ~3 years. Lower = earlier switch.
_YEAR_MAJOR_MIN_SPAN_DAYS = 1000
# Explicit observation ticks need more length at the standard wide, short figure
# size; keep majors visibly dominant without touching locator-based rulers.
_MULTI_DAY_MAJOR_TICK_LENGTH = 6
_MULTI_DAY_MINOR_TICK_LENGTH = 4


def _annotation_color(color):
    """
    Return a darker annotation colour for a segment fill colour.

    Annotation text and vertical dividers historically derived their colour with
    ``color[5:]``, which only produced a valid colour for the default ``light*``
    named colours (``lightgreen`` -> ``green``). Any other matplotlib colour
    format -- short/full hex, ``tab:*`` names, tuples, ``None`` -- raised or
    yielded nonsense. This preserves the exact legacy result for ``light*`` names
    (keeping default plots pixel-identical) and otherwise darkens the colour in
    RGBA space, which is well-defined for every valid matplotlib colour.

    Args:
        color: A matplotlib colour, or ``None``.

    Returns:
        A darker colour, or ``None`` (deferred to matplotlib's default).
    """
    if color is None:
        return None
    if isinstance(color, str) and color.startswith('light'):
        return color[5:]
    r, g, b, _ = mcolors.to_rgba(color)
    return (r * 0.7, g * 0.7, b * 0.7)


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

def _adjacent_to(index, value, offset):
    """
    Return the index value adjacent to *value* (``offset`` of +1 or -1), or None.

    Boundary displacement must follow the actual point spacing rather than a
    hard-coded daily step, otherwise non-daily data (e.g. weekly dates) leaves
    white gaps between shaded segments. For a contiguous daily index this is
    equivalent to ``value +/- 1 day``.
    """
    try:
        pos = index.get_loc(value)
    except KeyError:
        return None
    if not isinstance(pos, int):
        return None  # non-unique index: no well-defined adjacent point
    return _safe_adjacent(index, pos, offset)

def _thinned_positions(index, cap):
    """Observation positions thinned to at most *cap* ticks."""
    if len(index) <= cap:
        return index
    return index[::int(np.ceil(len(index) / cap))]

def _pinned_tick_positions(index):
    """Pick tick positions pinned to observations, thinned past the cap.

    Purely positional (``index[::ceil(n/_MAX_PINNED_TICKS)]``), so it is
    agnostic to index granularity: non-daily dates (weekly, fortnightly,
    month-end, yearly) and numerical indexes all follow the same rule. Used as
    the fallback when no calendar unit fits a multi-day cadence.
    """
    return _thinned_positions(index, _MAX_PINNED_TICKS)

def _sampling_locator(interval_days):
    """Locator marking every *interval_days*, or None when unexpressible.

    matplotlib's time locators align by minute-of-hour or hour-of-day, so a
    sampling gap is expressible only if it divides an hour (``MinuteLocator``)
    or a day (``HourLocator``, ``DayLocator``). 30-minute bars ->
    ``MinuteLocator(30)``, hourly -> ``HourLocator(1)``, daily ->
    ``DayLocator()``. A 45- or 90-minute gap has no such locator, so callers
    fall back to positional ticks.
    """
    minutes = interval_days * 24 * 60
    if abs(minutes - round(minutes)) > 1e-6:
        return None
    minutes = int(round(minutes))
    if minutes < 1:
        return None
    if minutes < 60:
        return mdates.MinuteLocator(interval=minutes) if 60 % minutes == 0 else None
    if minutes % 60:
        return None
    hours = minutes // 60
    if hours < 24:
        return mdates.HourLocator(interval=hours) if 24 % hours == 0 else None
    return mdates.DayLocator(interval=max(1, int(np.ceil(interval_days - 1e-9))))

def _minor_locator(step_days, span_steps, index):
    """True-granularity minor locator, thinned under matplotlib's tick limit.

    The interval tracks the data's own sampling gap. When the span projects more
    than ``_MAX_MINOR_TICKS`` of them the interval is coarsened by
    ``ceil(count / limit)``. matplotlib's hard ceiling is ``MAXTICKS`` (1000), but
    900 is the legibility target -- a ruler, not a band (e.g. 60 days of
    30-minute bars: 2880 -> interval x4 -> every 2 hours, ~720 ticks). Returns an
    explicit array of positions for a gap no locator can express (e.g. 45-minute
    bars), thinned to the same target.
    """
    base = _sampling_locator(step_days)
    if base is None:
        # Inexpressible cadence -> the true-granularity ruler as explicit
        # observation positions. Fixed ticks bypass Locator.MAXTICKS, so the
        # thinning here is our own legibility choice.
        return _thinned_positions(index, _MAX_MINOR_TICKS)
    if span_steps <= _MAX_MINOR_TICKS:
        return base
    # Coarsen to the first expressible multiple, so 30-minute bars thin to whole
    # hours rather than an unexpressible 5.5 hours (which would drop the ruler).
    factor = int(np.ceil(span_steps / _MAX_MINOR_TICKS))
    locator = _sampling_locator(step_days * factor)
    while locator is None:
        factor += 1
        locator = _sampling_locator(step_days * factor)
    return locator


def _divides(step_days, unit_days):
    """True if *step_days* is at most *unit_days* and divides it near-exactly."""
    if step_days <= 0 or step_days > unit_days:
        return False
    ratio = unit_days / step_days
    return abs(ratio - round(ratio)) < 1e-6

def _expressible_hour_interval(step_days, min_hours):
    """Smallest whole-hour interval >= *min_hours* an ``HourLocator`` can mark.

    ``HourLocator`` aligns to whole hours of the day, so only intervals dividing
    24 hours are expressible; the interval must also divide the sampling gap
    evenly so every observation lands on a major gridline. Returns the interval
    in hours, or None when no hour granularity fits (e.g. a 5-hour gap).
    """
    step_minutes = step_days * 24 * 60
    for hours in range(1, 24):
        if 24 % hours or hours < min_hours:
            continue
        ratio = hours * 60 / step_minutes
        if abs(ratio - round(ratio)) < 1e-6:
            return hours
    return None

def _intraday_major_locator(step_days, span_days):
    """Major locator for sub-daily data, widening with span; None if no unit fits.

    Each rung searches upward from its preferred interval for the smallest
    expressible multiple that also divides the sampling gap; only when none
    exists does the next rung get a turn, and the terminal fallback is positional
    ticks. A 30-minute gap gives 2-hour majors under a one-day span, 6-hour over
    three days, days thereafter; a 45-minute gap gives 3-hour majors under a day
    (the smallest whole-hour multiple of 45 minutes), then 6-hour over three
    days, and nothing expressible beyond -- where the caller pins positions.
    """
    if span_days <= 1:
        # Exactly-hourly keeps HourLocator(1); every other cadence starts at 2 h.
        floor = 1 if np.isclose(step_days, 1 / 24) else 2
        hours = _expressible_hour_interval(step_days, floor)
        if hours is not None:
            return mdates.HourLocator(interval=hours)
    if span_days <= 3:
        hours = _expressible_hour_interval(step_days, 6)
        if hours is not None:
            return mdates.HourLocator(interval=hours)
    if not _divides(step_days, 1):
        return None
    if span_days <= 7:
        return mdates.DayLocator()
    if span_days <= _WEEKLY_MAX_SPAN_DAYS:
        return mdates.WeekdayLocator(interval=1)
    if span_days <= _BIWEEKLY_MAX_SPAN_DAYS:
        return mdates.WeekdayLocator(interval=2)
    return mdates.MonthLocator()


def _format_for(major_locator):
    """Major label format keyed to the major unit, not the span.

    Hour-or-finer majors carry date and time on two lines (sub-day plots need
    the clock); day-or-coarser majors (Day/Weekday/Month locators, or pinned
    positions) read as dates only. So a 3-7-day span on the day rung no longer
    labels a trailing ``00:00``.
    """
    if isinstance(major_locator, (mdates.HourLocator, mdates.MinuteLocator,
                                  mdates.SecondLocator)):
        return '%Y-%m-%d\n%H:%M'
    return '%Y-%m-%d'


def _calendar_major_locator(step_days, span_days):
    """Calendar-unit majors for multi-day cadences, scaled to the label cap.

    Majors step up from the cadence to a calendar unit: weekly-ish cadences
    (< 15 days) read month by month; monthly-or-coarser cadences use year majors
    once the span reaches ~3 years (shorter monthly series stay month-based);
    yearly cadences always use year majors. The interval scales so the label
    count stays near ``_MAX_PINNED_TICKS``.
    """
    use_years = step_days >= 200 or (step_days >= 15 and span_days >= _YEAR_MAJOR_MIN_SPAN_DAYS)
    units = span_days / (365.25 if use_years else 30.44)
    interval = max(1, int(np.ceil(units / _MAX_PINNED_TICKS)))
    if use_years:
        return mdates.YearLocator(base=interval)
    return mdates.MonthLocator(interval=interval)


def _date_tick_spec(index, span_steps):
    """Pick date-axis tick strategy from sampling cadence and span.

    *span_steps* is the positional span ``len(index) - 1``. Cadence is read once
    from the median calendar gap (median is DST-robust for daily data):

    * Non-daily (gap > 1 day: weekly, fortnightly, month-end, yearly, ...):
      majors step up to a calendar unit -- month majors for weekly-ish
      cadences, year majors for monthly-or-coarser cadences once the span
      reaches ~3 years (yearly cadences always) -- scaled to keep labels near
      ``_MAX_PINNED_TICKS``. Minors are the data's own granularity as positional
      ticks at the observations, thinned to ``_MAX_MINOR_TICKS``. Positional
      pinning is the fallback when no calendar unit fits.
    * Daily (gap of one day): the original span-in-steps ladder -- weekly ->
      biweekly -> month majors, with a daily minor ruler.
    * Sub-daily: minors follow the true sampling granularity (30-minute bars ->
      every 30 minutes, hourly -> every hour), thinned below
      ``_MAX_MINOR_TICKS``; majors widen with the span in calendar days
      (hours -> 6 hours -> days -> weeks -> months). When no major unit divides
      the gap (45- or 90-minute bars past a few days), majors fall back to
      positional ticks; when no minor locator can express the gap, the minor slot
      becomes explicit positions at the observations.

    The one-year minor cut-off applies to the daily and sub-daily rulers only;
    multi-day minors thin (never drop) under ``_MAX_MINOR_TICKS``.

    Returns ``(major_locator, minor_locator, pinned_positions, date_format)``.
    The major slot is a locator or None; the minor slot is a locator, an array of
    positions, or None; locators and pinned positions are mutually exclusive.
    """
    # The one calendar read: the series' sampling cadence in days, taken as the
    # median calendar gap between observations. Median (not mean) so DST shifts
    # and occasional gaps don't skew it.
    diffs = np.diff(index.values)
    median_step_days = np.median(diffs) / np.timedelta64(1, 'D') if len(diffs) else 1
    span_days = span_steps * median_step_days

    if median_step_days > 1:
        major = _calendar_major_locator(median_step_days, span_days)
        if major is None:
            return None, None, _pinned_tick_positions(index), '%Y-%m-%d'
        return major, _thinned_positions(index, _MAX_MINOR_TICKS), None, '%Y-%m-%d'

    # Minors only while the span is within a year; past that the axis reads as a
    # calendar.
    minor = _minor_locator(median_step_days, span_steps, index) if span_days <= 365 else None

    if np.isclose(median_step_days, 1):
        if span_days <= _WEEKLY_MAX_SPAN_DAYS:
            return mdates.WeekdayLocator(interval=1), mdates.DayLocator(), None, '%Y-%m-%d'
        if span_days <= _BIWEEKLY_MAX_SPAN_DAYS:
            return mdates.WeekdayLocator(interval=2), mdates.DayLocator(), None, '%Y-%m-%d'
        # Beyond ~13 months: month majors with the year-capped minor ruler.
        return mdates.MonthLocator(), minor, None, '%Y-%m-%d'

    major = _intraday_major_locator(median_step_days, span_days)
    if major is None:
        return None, minor, _pinned_tick_positions(index), _format_for(major)
    return major, minor, None, _format_for(major)


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
    # 'date' and 'datetime64' are both real date axes: boundaries must be
    # displaced by the actual point spacing (via _adjacent_to), not a hard-coded
    # one-day step, or non-daily data leaves white gaps between shaded regions.
    is_date_axis = index_type in ('date', 'datetime64')
    for i, seg in enumerate(segments_enhanced):
        
        if is_date_axis:
            start = pd.to_datetime(seg['start'])
            end = pd.to_datetime(seg['end'])
        else:
            start = seg['start']
            end = seg['end']
        
        color = color_map.get(seg['direction'], 'gray')

        # Get context on prev seg if possible
        prev_seg = segments_enhanced[i-1] if i-1 >= 0 else None
        if is_date_axis:
            prev_point = _adjacent_to(df.index, start, -1)
            prev_neighbouring = prev_seg and (prev_point is not None) and (pd.to_datetime(prev_seg['end']) == prev_point)
        elif index_type == 'string':
            prev_neighbouring = prev_seg and (prev_seg['end'] == df.index[df.index.get_loc(start) - 1])
        else:
            prev_point = _adjacent_to(df.index, start, -1)
            prev_neighbouring = prev_seg and (prev_point is not None) and (prev_seg['end'] == prev_point)

        is_prev_not_trend = prev_seg and (not ('trend_class' in prev_seg))

        # Current seg context
        is_abrupt = ('trend_class' in seg and seg['trend_class'] == 'abrupt')
        is_noise = (seg['direction'] == 'Noise')
        is_not_trend = not ('trend_class' in seg)

        # Get context on next seg if possible
        next_seg = segments_enhanced[i+1] if i+1 < len(segments_enhanced) else None
        if is_date_axis:
            next_point = _adjacent_to(df.index, end, 1)
            next_neighbouring = next_seg and (next_point is not None) and (pd.to_datetime(next_seg['start']) == next_point)
        elif index_type == 'string':
            end_pos = df.index.get_loc(end)
            next_neighbouring = next_seg and (next_seg['start'] == _safe_adjacent(df.index, end_pos, 1))
        else:
            next_point = _adjacent_to(df.index, end, 1)
            next_neighbouring = next_seg and (next_point is not None) and (next_seg['start'] == next_point)
        
        next_seg_abrupt = next_seg and (('trend_class' in next_seg) and (next_seg['trend_class'] == 'abrupt'))
        next_seg_noise = next_seg and (next_seg['direction'] == 'Noise')

        # Adjust starts when appropriate
        if is_abrupt or is_noise: 
            pass  # Keep start as-is for abrupt/noise segments
        else: 
            if is_date_axis:
                new_start = _adjacent_to(df.index, start, -1) # Everything else displaced left start
            elif index_type == 'string':
                start_pos = df.index.get_loc(start)
                new_start = _safe_adjacent(df.index, start_pos, -1)
            else:
                new_start = _adjacent_to(df.index, start, -1) # Everything else displaced left start

            # Check validity of plot start adjustment
            value_new_start = df.loc[new_start, value_col] if new_start is not None and new_start in df.index else None

            value = df.loc[start, value_col]

            valid_up_start = (value_new_start) and (seg['direction'] == 'Up') and (value_new_start < value)
            valid_down_start = (value_new_start) and (seg['direction'] == 'Down') and (value_new_start > value)
            if (valid_up_start or valid_down_start or is_not_trend) and new_start is not None:
                start = new_start # Apply left displacement only if valid
            else: 
                # if not displaced and prev is not trend, adjust by plotting (as prev has already been drawn)
                if is_prev_not_trend and prev_neighbouring:
                    if is_date_axis:
                        prev_end = pd.to_datetime(segments_enhanced[i-1]['end'])
                        prev_adj_end = _adjacent_to(df.index, prev_end, 1)
                        prev_new_end = prev_adj_end.strftime('%Y-%m-%d') if prev_adj_end is not None else None
                    elif index_type == 'string':
                        prev_end = segments_enhanced[i-1]['end']
                        prev_end_pos = df.index.get_loc(prev_end)
                        prev_new_end = _safe_adjacent(df.index, prev_end_pos, 1)
                    else:
                        prev_end = segments_enhanced[i-1]['end']
                        prev_new_end = _adjacent_to(df.index, prev_end, 1)
                    
                    if prev_new_end is not None:
                        if index_type == 'string':
                            mask = (np.arange(len(df)) >= df.index.get_loc(prev_end)) & (np.arange(len(df)) <= df.index.get_loc(prev_new_end))
                        else:
                            mask = (df.index >= prev_end) & (df.index <= prev_new_end)
                        prev_color = color_map.get(segments_enhanced[i-1]['direction'], 'gray')
                        ax.fill_between(df.index[mask], ymin, ymax, color=prev_color, alpha=default_params['alpha'])

        # Adjust ends when appropriate
        if (next_seg_abrupt or next_seg_noise) and next_neighbouring:
            if is_date_axis:
                new_end = _adjacent_to(df.index, end, 1)
            elif index_type == 'string':
                end_pos = df.index.get_loc(end)
                new_end = _safe_adjacent(df.index, end_pos, 1)
            else:
                new_end = _adjacent_to(df.index, end, 1)
            
            # Check validity of plot end adjustment
            value_new_end = df.loc[new_end, value_col] if new_end is not None and new_end in df.index else None
            value = df.loc[end, value_col]

            valid_up_end = (value_new_end) and (seg['direction'] == 'Up') and (value_new_end > value)
            valid_down_end = (value_new_end) and (seg['direction'] == 'Down') and (value_new_end < value)
            is_not_trend = not ('trend_class' in seg)
            if (valid_up_end or valid_down_end or is_not_trend) and new_end is not None:
                end = new_end  # Apply right displacement only if valid
            else: 
                # if not displaced and next is noise, adjust for next plotting round
                if next_seg_noise and next_neighbouring: 
                    if is_date_axis:
                        next_start = pd.to_datetime(segments_enhanced[i+1]['start'])
                        next_adj_start = _adjacent_to(df.index, next_start, -1)
                        if next_adj_start is not None:
                            segments_enhanced[i+1]['start'] = next_adj_start.strftime('%Y-%m-%d')
                    elif index_type == 'string':
                        next_start_pos = df.index.get_loc(segments_enhanced[i+1]['start'])
                        segments_enhanced[i+1]['start'] = _safe_adjacent(df.index, next_start_pos, -1)
                    else:
                        next_start = segments_enhanced[i+1]['start']
                        next_adj_start = _adjacent_to(df.index, next_start, -1)
                        if next_adj_start is not None:
                            segments_enhanced[i+1]['start'] = next_adj_start
        else: 
            pass  # Keep end as-is

        if index_type == 'string':
            mask = (np.arange(len(df)) >= df.index.get_loc(start)) & (np.arange(len(df)) <= df.index.get_loc(end))
        else:
            mask = (df.index >= start) & (df.index <= end) 


        ax.fill_between(df.index[mask], ymin, ymax, color=color, alpha=default_params['alpha'])
        
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
                    color=_annotation_color(color))
            
        # Add vertical line if next seg is same & touching
        if next_seg and next_neighbouring and next_seg['direction'] == seg['direction']:
            if is_date_axis:
                line_date = pd.to_datetime(seg['end'])
            else:
                line_date = seg['end']
            ax.axvline(x=line_date, color=_annotation_color(color), linewidth=0.5)

    # Set limits
    if index_type == 'string':
        first_date = df.index[0]
        last_date = df.index[-1]
    else:
        first_date = df.index.min()
        last_date = df.index.max()

    ax.set_xlim(first_date, last_date)
    ax.set_ylim(ymin, ymax)

    if index_type in ('date', 'datetime64'):
        index = df.index
        span_steps = len(index) - 1

        major, minor, pinned, date_format = _date_tick_spec(index, span_steps)
        if pinned is not None:
            # Positional ticks: pin majors to the data's own index positions so
            # every major (and its 'major' gridline) lands exactly on an
            # observation, for any cadence including irregular. A WeekdayLocator
            # anchors its interval grid to the Unix epoch, which lands majors
            # beside fortnightly / month-end observations.
            ax.set_xticks(pinned)
        else:
            # Locator-based majors: coarsen with span so long series avoid a
            # wall of labels. Minors carry the data's own sampling granularity
            # where a matplotlib locator can express it, else they are omitted.
            ax.xaxis.set_major_locator(major)
        if minor is not None:
            if isinstance(minor, (np.ndarray, pd.Index)):
                # Inexpressible cadence: explicit positional ruler, bypassing
                # Locator.MAXTICKS.
                ax.set_xticks(minor, minor=True)
                is_multi_day = (len(index) > 1 and
                                np.median(np.diff(index.values)) / np.timedelta64(1, 'D') > 1)
                if is_multi_day:
                    ax.tick_params(axis='x', which='major', length=_MULTI_DAY_MAJOR_TICK_LENGTH,
                                   width=1.0)
                    ax.tick_params(axis='x', which='minor', length=_MULTI_DAY_MINOR_TICK_LENGTH,
                                   width=0.8)
            else:
                ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

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
