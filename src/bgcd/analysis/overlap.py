# ================================
# File: src/bgcd/analysis/overlap.py
# ================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class OverlapWindow:
    """
    A continuous time window where all required variables are available (non-NaN).
    """
    start: pd.Timestamp
    end: pd.Timestamp
    n_rows: int
    duration_h: float


@dataclass
class OverlapResult:
    """
    Results of overlap/window detection.
    """
    required: list[str]
    optional: list[str]
    windows: list[OverlapWindow]
    best: Optional[OverlapWindow]


def find_overlap_windows(
    df: pd.DataFrame,
    *,
    time_col: str = "time_utc",
    required: Iterable[str],
    optional: Iterable[str] = (),
    min_points: int = 50,
    max_gap_h: float = 6.0,
    prefer: str = "duration",  # "duration" or "n_rows"
) -> OverlapResult:
    """
    Find continuous windows where all 'required' variables are simultaneously available.

    Definitions:
    - A row is "valid" if all required variables are non-NaN.
    - A window is a maximal consecutive sequence of valid rows, allowing time gaps <= max_gap_h.
      (So if there's a bigger temporal gap, we break the window even if values are valid.)

    Params:
    - min_points: discard windows with fewer rows
    - max_gap_h: split a window if dt > max_gap_h
    - prefer: how to choose the "best" window ("duration" or "n_rows")

    Notes:
    - Optional variables do NOT affect window validity.
      They are only carried in metadata for traceability.
    """
    req = [str(v) for v in required if str(v) in df.columns]
    opt = [str(v) for v in optional if str(v) in df.columns]

    if time_col not in df.columns:
        raise ValueError(f"Missing {time_col!r} column.")
    if not req:
        raise ValueError("No required variables found in dataframe columns.")

    # time sorted
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    out = out.loc[out[time_col].notna()].sort_values(time_col)

    # valid rows mask: all required non-NaN
    valid = pd.Series(True, index=out.index)
    for v in req:
        valid &= out[v].notna()

    if not valid.any():
        return OverlapResult(required=req, optional=opt, windows=[], best=None)

    t = out[time_col].to_numpy(dtype="datetime64[ns]")
    dt_h = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0
    # gap_break[i] refers to break between i and i+1
    gap_break = np.zeros(len(dt_h), dtype=bool)
    gap_break[dt_h > float(max_gap_h)] = True

    # Build windows over the "valid" sequence, also split on gap_break
    idx = out.index.to_numpy()
    valid_np = valid.to_numpy()

    windows: list[OverlapWindow] = []

    start_i: Optional[int] = None
    for i in range(len(idx)):
        is_valid = bool(valid_np[i])

        # determine if we should break before this point (i) due to previous gap
        if i > 0 and gap_break[i - 1]:
            # close an open window if any
            if start_i is not None:
                end_i = i - 1
                windows.append(_make_window(out, time_col, start_i, end_i))
                start_i = None

        if is_valid and start_i is None:
            start_i = i
        elif (not is_valid) and start_i is not None:
            end_i = i - 1
            windows.append(_make_window(out, time_col, start_i, end_i))
            start_i = None

    # close at end
    if start_i is not None:
        windows.append(_make_window(out, time_col, start_i, len(idx) - 1))

    # filter by min_points
    windows = [w for w in windows if w.n_rows >= int(min_points)]

    best: Optional[OverlapWindow] = None
    if windows:
        prefer = str(prefer).lower()
        if prefer == "n_rows":
            best = max(windows, key=lambda w: (w.n_rows, w.duration_h))
        else:
            best = max(windows, key=lambda w: (w.duration_h, w.n_rows))

    return OverlapResult(required=req, optional=opt, windows=windows, best=best)


def _make_window(df: pd.DataFrame, time_col: str, start_i: int, end_i: int) -> OverlapWindow:
    t0 = df[time_col].iloc[start_i]
    t1 = df[time_col].iloc[end_i]
    n = int(end_i - start_i + 1)
    dur_h = float((t1 - t0).total_seconds() / 3600.0) if pd.notna(t0) and pd.notna(t1) else np.nan
    return OverlapWindow(start=t0, end=t1, n_rows=n, duration_h=dur_h)