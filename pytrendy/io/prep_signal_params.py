"""**Signal-Parameter Preparation**

The signal-processing stages share a common set of constants — smoothing and
rolling windows, grouping distance, and the noise/flat thresholds. ``detect_trends``
is the single public entry point, so this module is the one place those constants
are defaulted before the pipeline runs.

``window_smooth`` is granularity-aware: when the caller does not override it, it is
inferred from the index cadence so a single smoothing window spans a comparable
stretch of the series at every sampling frequency. The inferred point counts are:

| Cadence     | P (pts/cycle) | window_smooth (= 2P+1) |
| ----------- | ------------- | ---------------------- |
| 15 minutes  | 96            | 193                    |
| 30 minutes  | 48            | 97                     |
| hourly      | 24            | 49                     |
| daily       | 7             | 15                     |
| weekly      | 4             | 9                      |
| monthly     | 12            | 25                     |
| quarterly   | 4             | 9                      |
| yearly      | — (fallback)  | 5                      |
| other       | —             | 15                     |

The table entries satisfy ``int(window_smooth × 0.5) = P``, so the default
``smooth_factor``/``noise_factor`` (0.5) make ``window_flat``/``window_noise``
span one seasonal cycle while ``window_smooth`` spans two. Keep new rows on the
``2P+1`` ladder.

The table is matched pragmatically against :func:`pandas.infer_freq` labels:
``T``/``min`` aliases containing 15 or 30, hourly ``H``/``h``, daily ``D``,
weekly ``W...``, monthly ``M``/``ME``/``MS``,
quarterly ``Q...`` and yearly ``Y``/``A``/``YE``/``YS``. An explicit user ``window_smooth`` always wins and
skips inference. The resolved window is finally clamped to the series length
(decrementing by two to stay odd) with a floor of 3, so Savitzky-Golay always
receives a valid window.

Defaults are merged with any user-supplied ``signal_params`` overrides. The derived
windows ``window_flat`` and ``window_noise`` are not top-level defaults: unless
explicitly overridden they derive from ``window_smooth``. Every stage downstream
assumes :func:`prep_signal_params` returns a fully-populated dict.
"""

import warnings

import numpy as np
import pandas as pd

_SIGNAL_PARAMS_DEFAULTS = {
    'window_smooth': 15,        # Savitzky-Golay smoothing window, in points.
    'smooth_factor': 0.5,       # Fraction of window_smooth spanned by window_flat: raise to widen it
                                # (smoother flat baseline, fewer flags), lower to narrow it (more sensitive);
                                # scaled to window_smooth so it stays proportional. Historical name: it scales
                                # window_flat, not smoothing; with the 2P+1 table it recovers P.
    'noise_factor': 0.5,        # Fraction of window_smooth spanned by window_noise: raise to widen it (smoother SNR estimate, fewer flags), lower to narrow it (more sensitive); scaled to window_smooth so it stays proportional.
    'grouping_distance': 7,     # Maximum gap, in index steps, for grouping nearby segments.
    'min_trend_length': 3,      # Minimum length, in steps, for an Up/Down segment to be retained.
    'min_flat_noise_length': 1, # Minimum length, in steps, for a Flat/Noise segment to be retained.
    'threshold_noise': 2.5,     # SNR threshold (dB) below which a region is classified as noise.
    'threshold_smooth': 0.001,  # Derivative threshold as a fraction of the signal IQR, below which motion counts as flat.
    'threshold_flat': 0.835,    # Flat sensitivity as a fraction of the minimum non-zero rolling std.
}

# Inferred smoothing window per cadence label. See the module docstring for the table.
_WINDOW_BY_CADENCE = {
    '15min': 193,
    '30min': 97,
    'H': 49,
    'D': 15,
    'W': 9,
    'ME': 25,
    'QE': 9,
    'YE': 5,
}
_DEFAULT_WINDOW = _SIGNAL_PARAMS_DEFAULTS['window_smooth']


def _clamp_window(window: int, n: int|None) -> int:
    """Shrink ``window`` to the series length, keeping it odd with a floor of 3."""
    if n is None:
        return window
    while window > n:
        window -= 2
    return max(window, 3)


def _as_datetime_index(index):
    """Return a ``DatetimeIndex`` view of ``index`` when it is date-like, else ``None``.

    Numeric indexes are never coerced: integer/float values would otherwise be read
    as epoch timestamps by :func:`pandas.to_datetime`. Only datetime dtypes and
    genuinely parseable string/object labels are accepted.
    """
    if index is None:
        return None
    try:
        if isinstance(index, pd.DatetimeIndex):
            return index
        if isinstance(index, pd.Series):
            dtype = index.dtype
        else:
            arr = np.asarray(index)
            if arr.ndim == 0:
                return None
            if np.issubdtype(arr.dtype, np.datetime64):
                return pd.DatetimeIndex(arr)
            dtype = arr.dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.DatetimeIndex(index)
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not infer format.*")
                parsed = pd.to_datetime(index, errors='coerce')
            if parsed.notna().all():
                return pd.DatetimeIndex(parsed)
    except (TypeError, ValueError):
        return None
    return None


def _window_from_label(label: str|None) -> int|None:
    """Map a ``pandas.infer_freq`` label to its smoothing window, or ``None`` if unmapped."""
    if label is None:
        return None
    lab = str(label)
    low = lab.lower()

    # Sub-hourly: 'T'/'min' aliases (e.g. '15T', '30min'); other spans are unmapped.
    if 'min' in low or low.endswith('t'):
        if '15' in lab:
            return _WINDOW_BY_CADENCE['15min']
        if '30' in lab:
            return _WINDOW_BY_CADENCE['30min']
        return None
    if low == 'h':                      # hourly ('H'/'h'); '5H' etc. stay unmapped
        return _WINDOW_BY_CADENCE['H']
    if low == 'd':                      # daily
        return _WINDOW_BY_CADENCE['D']
    if low.startswith('w'):             # weekly ('W', 'W-SUN', ...)
        return _WINDOW_BY_CADENCE['W']
    if low.startswith('m'):             # monthly ('M'/'ME'/'MS'); 'min' handled above
        return _WINDOW_BY_CADENCE['ME']
    if low.startswith('q'):             # quarterly
        return _WINDOW_BY_CADENCE['QE']
    if low.startswith('y') or low.startswith('a'):  # yearly ('Y'/'A'/'YE'/'YS')
        return _WINDOW_BY_CADENCE['YE']
    return None


def infer_window_smooth(index, n: int|None=None) -> int:
    """
    Infer a smoothing window from the sampling cadence of an index.

    Converts ``index`` to a ``DatetimeIndex`` (date-like values only), asks
    :func:`pandas.infer_freq` for its frequency label, and maps that label through
    the cadence table. Anything unmapped — a non-datetime index, an irregular or
    unparseable one, or an exotic frequency such as ``'45T'``/``'5H'``/``'14D'`` —
    falls back to 15. The result is clamped to ``n`` when provided.

    Args:
        index: External index values (pandas Index/Series, or array-like).
        n (int, optional): Series length used to clamp the inferred window.

    Returns:
        int: Inferred smoothing window (clamped to ``n`` when given).
    """
    window = _DEFAULT_WINDOW
    dt_index = _as_datetime_index(index)
    if dt_index is not None and len(dt_index) >= 3:
        try:
            freq = pd.infer_freq(dt_index)
        except ValueError:
            freq = None
        label_window = _window_from_label(freq)
        if label_window is not None:
            window = label_window
    return _clamp_window(window, n)


def prep_signal_params(signal_params: dict|None=None, index=None, n: int|None=None) -> dict:
    """
    Merge user-supplied `signal_params` over the defaults and derive window sizes.

    `window_smooth` is inferred from `index`'s cadence when the user does not
    explicitly set it; an explicit value always wins. The resolved window is then
    clamped to the series length. `window_flat` and `window_noise` are not
    top-level defaults: unless explicitly overridden they derive from the clamped
    `window_smooth`. Every stage downstream assumes this returns a fully-populated
    dict.

    Args:
        signal_params (dict, optional): User overrides. Unknown keys are forwarded silently.
        index (optional): External index; its cadence infers `window_smooth` when the
            user does not supply one.
        n (int, optional): Series length used to clamp `window_smooth`. Defaults to
            `len(index)` when `index` is given.

    Returns:
        dict: Fully-populated signal_params with every key the stages read.
    """
    user_params = signal_params or {}
    resolved = {**_SIGNAL_PARAMS_DEFAULTS, **user_params}

    if n is None and index is not None:
        n = len(index)

    if 'window_smooth' in user_params:
        # Explicit value wins; still clamp it to a Savitzky-Golay-valid window.
        resolved['window_smooth'] = _clamp_window(resolved['window_smooth'], n)
    elif index is not None:
        resolved['window_smooth'] = infer_window_smooth(index, n)
    else:
        resolved['window_smooth'] = _clamp_window(resolved['window_smooth'], n)

    # Derive the factor-based windows *after* clamping, and only when the user did
    # not set them explicitly.
    resolved.setdefault('window_flat', int(resolved['window_smooth'] * resolved['smooth_factor']))
    resolved.setdefault('window_noise', int(resolved['window_smooth'] * resolved['noise_factor']))
    return resolved
