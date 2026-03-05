# ================================
# File: src/bgcd/analysis/lag_corr.py
# ================================
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from scipy.stats import pearsonr


@dataclass
class LagCorrResult:
    pair_name: str
    df: pd.DataFrame  # columns: lag_hours, r, n_pairs


@dataclass
class LagCorrResult:
    pair_name: str
    df: pd.DataFrame  # columns: lag_hours, r, p_value, n_pairs



def lag_correlation_asof(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    time_col: str = "time_utc",
    max_lag_hours: int = 72,
    lag_step_hours: int = 3,
    match_tolerance_hours: float = 1.5,
    min_pairs: int = 30,
) -> LagCorrResult:
    """
    Lagged correlation corr(x(t), y(t+lag)) computed without interpolation using merge_asof.

    Convention:
      - lag > 0 means y is evaluated AFTER x (y lags x), i.e. x leads.
      - For each lag, we match x times to shifted y times within tolerance.

    Returns:
      DataFrame with lag_hours, r, n_pairs.
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Missing columns: {x}, {y}")

    d = df[[time_col, x, y]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)
    d = d.dropna(subset=[time_col]).sort_values(time_col)

    dx = d[[time_col, x]].dropna().rename(columns={x: "x"}).sort_values(time_col)
    dy = d[[time_col, y]].dropna().rename(columns={y: "y"}).sort_values(time_col)

    tol = pd.Timedelta(hours=float(match_tolerance_hours))

    lags = list(range(-int(max_lag_hours), int(max_lag_hours) + 1, int(lag_step_hours)))
    rows = []

    for lag_h in lags:
        lag = pd.Timedelta(hours=int(lag_h))

        # We want corr(x(t), y(t+lag_h)).
        # Shift y times backward by lag so that y(t+lag) aligns with x(t).
        dy_shift = dy.copy()
        dy_shift[time_col] = dy_shift[time_col] - lag
        dy_shift = dy_shift.sort_values(time_col)

        m = pd.merge_asof(
            dx,
            dy_shift,
            on=time_col,
            direction="nearest",
            tolerance=tol,
        ).dropna(subset=["x", "y"])

        n = int(len(m))
        if n >= int(min_pairs):
            xv = m["x"].to_numpy(dtype=float)
            yv = m["y"].to_numpy(dtype=float)
            r, p = pearsonr(xv, yv)
        else:
            r = np.nan
            p = np.nan

        rows.append({"lag_hours": lag_h, "r": r, "p_value": p, "n_pairs": n})

    out = pd.DataFrame(rows)
    return LagCorrResult(pair_name=f"{x}__{y}", df=out)