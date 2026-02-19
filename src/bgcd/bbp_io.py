# src/bgcd/bbp_io.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

from .oxygen_io import matlab_datenum_to_datetime  # reuse existing, already correct

# input columns (as seen in your bbp_raw csv)
RENAME = {
    "t": "time_raw",
    "lat": "lat",
    "lon": "lon",
    "bbp_1": "bbp_470_m1",  # 1 -> 470 nm
    "bbp_2": "bbp_532_m1",  # 2 -> 532 nm
}

REQUIRED = ["lat", "lon", "t", "bbp_1", "bbp_2"]


def platform_id_from_filename(p: str | Path) -> str | None:
    p = Path(p)
    m = re.search(r"(\d{10,})", p.name)
    return m.group(1) if m else None


def bbp_csv_to_dataframe(path: str | Path, *, platform_id: str | None = None) -> pd.DataFrame:
    """
    Read one bbp raw CSV (e.g. 300534065378180.csv) and return canonical dataframe:
      platform_id, time_utc, lat, lon, bbp_470_m1, bbp_532_m1

    Assumes t is MATLAB datenum (days).
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}. Available: {list(df.columns)}")

    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})

    # time conversion (MATLAB datenum)
    df.insert(0, "time_utc", matlab_datenum_to_datetime(df["time_raw"]))
    df = df.drop(columns=["time_raw"])

    # platform_id
    pid = str(platform_id) if platform_id is not None else (platform_id_from_filename(path) or "")
    df.insert(0, "platform_id", pid)

    # numeric coercion
    for c in ["lat", "lon", "bbp_470_m1", "bbp_532_m1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    df = df.dropna(subset=["platform_id", "time_utc", "lat", "lon"]).reset_index(drop=True)
    df = df.sort_values(["platform_id", "time_utc"]).reset_index(drop=True)
    return df
