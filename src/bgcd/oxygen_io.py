from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .mat_io import load_mat_any


def matlab_datenum_to_datetime(dn: pd.Series) -> pd.Series:
    """Convert MATLAB datenum (days) to pandas datetime (UTC-naive)."""
    from datetime import datetime, timedelta
    import numpy as np

    arr = pd.to_numeric(dn, errors="coerce").to_numpy(dtype=float)
    out = []
    for x in arr:
        if not np.isfinite(x):
            out.append(pd.NaT)
            continue
        day = int(x)
        frac = x - day
        try:
            dt = datetime.fromordinal(day) + timedelta(days=frac) - timedelta(days=366)
        except Exception:
            dt = None
        out.append(dt)
    return pd.to_datetime(out, errors="coerce")


def oxygen_mat_to_dataframe(
    mat_path: str | Path,
    *,
    platform_id: str,
    keep_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Load oxygen_*.mat and return a DataFrame with at least:
      platform_id, time_utc, lat, lon
    plus all requested keep_cols (if present).
    """
    d = load_mat_any(mat_path)

    # required keys in your files
    required = {
        "time_c": "time_c",
        "lat": "Lat_sub_c",
        "lon": "Lon_sub_c",
    }
    missing = [k for k, v in required.items() if v not in d]
    if missing:
        raise RuntimeError(f"{mat_path}: missing keys {missing}. Available: {sorted(d.keys())}")

    df = pd.DataFrame(
        {
            "platform_id": str(platform_id),
            "time_utc": matlab_datenum_to_datetime(pd.Series(d["time_c"])),
            "lat": pd.Series(d["Lat_sub_c"]).astype("float64", errors="ignore"),
            "lon": pd.Series(d["Lon_sub_c"]).astype("float64", errors="ignore"),
        }
    )

    # optional columns
    if keep_cols:
        for c in keep_cols:
            if c in ("platform_id", "time_utc", "lat", "lon"):
                continue
            if c in d:
                df[c] = pd.Series(d[c])

    # numeric coercion (only for non-key cols)
    for c in df.columns:
        if c in ("platform_id", "time_utc"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time_utc", "lat", "lon"]).reset_index(drop=True)
    return df


def preprocess_oxygen_df(
    df: pd.DataFrame,
    *,
    zero_is_nan_cols: Iterable[str] = ("DO2_c", "oxy_comp_mgL_c"),
) -> pd.DataFrame:
    """
    Minimal, non-invasive QC:
    - convert zeros to NaN for specified cols
    - drop rows without time/lat/lon
    """
    out = df.copy()

    for c in zero_is_nan_cols:
        if c in out.columns:
            out.loc[out[c] == 0, c] = pd.NA

    out = out.dropna(subset=["time_utc", "lat", "lon"]).reset_index(drop=True)
    return out
