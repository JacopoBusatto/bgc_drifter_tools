from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .io import read_drifter_csv, read_wind_csv, merge_drifter_wind


@dataclass(frozen=True)
class MasterPaths:
    drifter_csv: str | Path
    wind_csv: str | Path
    mat_db_dir: str | Path  # directory containing mat_timeseries_<id>.csv/.parquet


def _read_mat_timeseries(mat_db_dir: str | Path, platform_id: str) -> pd.DataFrame:
    """
    Read per-platform MAT-derived timeseries from db_mat.
    Accepts either CSV or Parquet (prefers Parquet if both exist).
    """
    d = Path(mat_db_dir)
    base = d / f"mat_timeseries_{platform_id}"

    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")

    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv, parse_dates=["time_utc"])

    raise FileNotFoundError(f"MAT timeseries not found for {platform_id}: {parquet} or {csv}")


def _merge_asof_per_platform(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: str,
    suffix: str,
) -> pd.DataFrame:
    """
    Nearest-time merge_asof of 'right' onto 'left', per platform_id.
    right must contain: platform_id, time_utc, ...
    """
    l = left.sort_values(["platform_id", "time_utc"]).copy()
    r = right.sort_values(["platform_id", "time_utc"]).copy()

    # avoid duplicate column names (except join keys)
    r_cols = [c for c in r.columns if c not in ("platform_id", "time_utc")]
    r_ren = {c: f"{c}{suffix}" for c in r_cols if c in l.columns}
    if r_ren:
        r = r.rename(columns=r_ren)

    out = []
    for pid, lpid in l.groupby("platform_id", sort=False):
        rpid = r[r["platform_id"] == pid]
        if rpid.empty:
            out.append(lpid)
            continue

        merged = pd.merge_asof(
            lpid,
            rpid.drop(columns=["platform_id"]),
            on="time_utc",
            direction="nearest",
            tolerance=pd.Timedelta(tolerance),
        )
        out.append(merged)

    return pd.concat(out, ignore_index=True)


def build_master_for_platform(
    platform_id: str,
    paths: MasterPaths,
    *,
    target_time: str = "drifter",      # "drifter" | "hourly"
    wind_tolerance: str = "30min",
    mat_tolerance: str = "30min",
    hourly_freq: str = "1H",
) -> pd.DataFrame:
    """
    Build a per-platform master table by merging:
    - drifter core data
    - wind stats
    - vorticity/strain MAT timeseries

    Parameters
    ----------
    target_time:
        "drifter" -> keep drifter timestamps; attach wind + mat via nearest-time merge_asof
        "hourly"  -> create hourly timeline from mat (or from drifter range) and attach others to it
    """
    pid = str(platform_id)

    d_all = read_drifter_csv(paths.drifter_csv)
    w_all = read_wind_csv(paths.wind_csv)
    m_all = _read_mat_timeseries(paths.mat_db_dir, pid)

    d = d_all[d_all["platform_id"] == pid].copy()
    w = w_all[w_all["platform_id"] == pid].copy()
    m = m_all[m_all["platform_id"] == pid].copy() if "platform_id" in m_all.columns else m_all.copy()
    if "platform_id" not in m.columns:
        m.insert(0, "platform_id", pid)

    if d.empty:
        raise ValueError(f"No drifter data found for platform_id={pid}")
    if m.empty:
        raise ValueError(f"No MAT timeseries found for platform_id={pid}")

    if target_time not in ("drifter", "hourly"):
        raise ValueError("target_time must be 'drifter' or 'hourly'")

    if target_time == "drifter":
        base = d.sort_values("time_utc").reset_index(drop=True)

        # wind onto drifter
        base = merge_drifter_wind(base, w, tolerance=wind_tolerance)

        # mat onto drifter
        base = _merge_asof_per_platform(base, m, tolerance=mat_tolerance, suffix="_mat")

        return base.reset_index(drop=True)

    # target_time == "hourly"
    # Use MAT time span as reference (already hourly-ish)
    tmin = m["time_utc"].min()
    tmax = m["time_utc"].max()
    timeline = pd.date_range(tmin.floor("H"), tmax.ceil("H"), freq=hourly_freq)

    base = pd.DataFrame({"time_utc": timeline})
    base.insert(0, "platform_id", pid)

    # attach MAT (usually already hourly; use tight tolerance like 5min if you want)
    base = _merge_asof_per_platform(base, m, tolerance=mat_tolerance, suffix="_mat")

    # attach drifter onto hourly (nearest)
    base = _merge_asof_per_platform(base, d, tolerance="2H", suffix="_dr")

    # attach wind onto hourly
    base = _merge_asof_per_platform(base, w, tolerance="2H", suffix="_w")

    return base.reset_index(drop=True)
