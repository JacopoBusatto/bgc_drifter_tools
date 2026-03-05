# ================================
# File: src/bgcd/analysis/rolling_corr.py
# (manual rolling, robust to pandas freq parsing issues)
# ================================
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class RollingCorrSeries:
    pair_name: str
    df: pd.DataFrame  # columns: time_utc, r, n_pairs


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.sqrt((x0 * x0).sum()) * np.sqrt((y0 * y0).sum())
    if den == 0:
        return np.nan
    return float((x0 * y0).sum() / den)


def rolling_corr_pair(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    time_col: str = "time_utc",
    window_hours: int = 48,
    min_points: int = 10,
    center: bool = True,
) -> RollingCorrSeries:
    """
    Rolling Pearson correlation on irregular time axis, WITHOUT pandas time-based rolling.

    For each timestamp t_i:
      - take points in [t_i - window/2, t_i + window/2] if center=True
        else take points in [t_i - window, t_i]
      - compute Pearson r on valid (x,y) pairs if n_pairs >= min_points

    Output:
      time_utc, r, n_pairs
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Missing columns: {x}, {y}")

    d = df[[time_col, x, y]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)
    d = d.dropna(subset=[time_col]).sort_values(time_col)

    t = d[time_col].to_numpy()
    xv = d[x].to_numpy()
    yv = d[y].to_numpy()

    # Precompute valid mask
    valid_xy = pd.notna(xv) & pd.notna(yv)

    # Window bounds
    W = pd.Timedelta(hours=int(window_hours))
    half = W / 2

    out_t = []
    out_r = []
    out_n = []

    # We'll use integer indexing; create a pandas Series of times for comparisons
    t_series = pd.Series(d[time_col].to_list())

    for i, ti in enumerate(t_series):
        if center:
            t0 = ti - half
            t1 = ti + half
        else:
            t0 = ti - W
            t1 = ti

        in_win = (t_series >= t0) & (t_series <= t1)
        ok = in_win.to_numpy() & valid_xy

        n = int(ok.sum())
        if n >= int(min_points):
            r = _pearson_r(xv[ok].astype(float), yv[ok].astype(float))
        else:
            r = np.nan

        out_t.append(ti)
        out_r.append(r)
        out_n.append(n)

    out = pd.DataFrame({"time_utc": out_t, "r": out_r, "n_pairs": out_n})
    return RollingCorrSeries(pair_name=f"{x}__{y}", df=out)


def rolling_corr_many(
    df: pd.DataFrame,
    *,
    pairs: List[Tuple[str, str]],
    time_col: str = "time_utc",
    window_hours: int = 48,
    min_points: int = 10,
    center: bool = True,
) -> List[RollingCorrSeries]:
    out: List[RollingCorrSeries] = []
    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            continue
        out.append(
            rolling_corr_pair(
                df,
                x=x,
                y=y,
                time_col=time_col,
                window_hours=int(window_hours),
                min_points=int(min_points),
                center=center,
            )
        )
    return out