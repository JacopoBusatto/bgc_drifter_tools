from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


# -----------------------------------------------------------------------------
# Paths & generic per-platform readers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MasterPaths:
    drifter_db_dir: str | Path   # contains drifter_<pid>.csv|parquet (canonical)
    wind_db_dir: str | Path      # contains wind_<pid>.csv|parquet (canonical)
    mat_db_dir: str | Path       # contains mat_timeseries_<pid>.csv|parquet
    extra_db_dirs: dict[str, str | Path] | None = None
    # extra_db_dirs example:
    # {"oxygen": "C:/.../db_oxygen", "chl": "C:/.../db_chl"}


def _read_per_platform(
    db_dir: str | Path,
    base_name: str,
    platform_id: str,
) -> pd.DataFrame:
    """
    Read a per-platform file either CSV or Parquet (prefers Parquet if both exist).
    Expects the filename pattern: {base_name}_{platform_id}.csv|parquet
    """
    d = Path(db_dir)
    base = d / f"{base_name}_{platform_id}"

    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")

    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        # parse_dates only if present
        df = pd.read_csv(csv)
        if "time_utc" in df.columns:
            df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
        return df

    raise FileNotFoundError(f"Missing per-platform file for {platform_id}: {parquet} or {csv}")


def _ensure_keys(df: pd.DataFrame, platform_id: str) -> pd.DataFrame:
    """
    Ensure platform_id and time_utc columns exist and are well-typed.
    """
    df = df.copy()

    if "platform_id" not in df.columns:
        df.insert(0, "platform_id", str(platform_id))
    df["platform_id"] = df["platform_id"].astype(str).str.strip()

    if "time_utc" not in df.columns:
        raise ValueError("Expected 'time_utc' column in per-platform DB.")
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    df = df.dropna(subset=["platform_id", "time_utc"])
    return df


# -----------------------------------------------------------------------------
# Merge helpers
# -----------------------------------------------------------------------------
def _merge_asof_single_platform(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: str,
    *,
    suffix: str = "",
    direction: str = "nearest",
) -> pd.DataFrame:
    """
    Nearest-time merge_asof for a single platform (platform_id already filtered).
    Keeps all left rows.
    """
    l = left.sort_values("time_utc").copy()
    r = right.sort_values("time_utc").copy()

    # avoid duplicate column names (except keys)
    r_cols = [c for c in r.columns if c not in ("platform_id", "time_utc")]
    ren = {c: f"{c}{suffix}" for c in r_cols if c in l.columns}
    if ren:
        r = r.rename(columns=ren)

    merged = pd.merge_asof(
        l,
        r.drop(columns=["platform_id"]),
        on="time_utc",
        direction=direction,
        tolerance=pd.Timedelta(tolerance),
    )
    return merged


def _select_columns(df: pd.DataFrame, keep: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in keep if c in df.columns]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        # We do NOT error here because schemas can evolve; just keep what's available.
        # If you prefer strictness, switch to raising.
        pass
    return df[cols].copy()

# -----------------------------------------------------------------------------
# Filter helpers
# -----------------------------------------------------------------------------
BBOX = (-10.0, 36.0, 30.0, 46.0)  # lon_min, lon_max, lat_min, lat_max


def filter_bbox_keep_only(d: pd.DataFrame, bbox=BBOX) -> pd.DataFrame:
    if d.empty:
        return d
    lon_min, lon_max, lat_min, lat_max = bbox
    m = d["lon"].between(lon_min, lon_max) & d["lat"].between(lat_min, lat_max)
    return d.loc[m].copy().reset_index(drop=True)


def filter_bbox_first_entry(d: pd.DataFrame, bbox=BBOX) -> pd.DataFrame:
    if d.empty:
        return d
    d = d.sort_values("time_utc").copy()
    lon_min, lon_max, lat_min, lat_max = bbox
    inside = d["lon"].between(lon_min, lon_max) & d["lat"].between(lat_min, lat_max)
    if not inside.any():
        return d.iloc[0:0].copy()
    t0 = d.loc[inside, "time_utc"].iloc[0]
    return d[d["time_utc"] >= t0].reset_index(drop=True)


def filter_largest_contiguous_segment(
    d: pd.DataFrame,
    *,
    max_gap: str = "7D",
) -> pd.DataFrame:
    """
    Keep only the largest contiguous time segment (by number of rows),
    splitting segments when time gaps exceed max_gap.
    """
    if d.empty:
        return d
    d = d.sort_values("time_utc").copy()
    dt = d["time_utc"].diff()
    seg_id = (dt > pd.Timedelta(max_gap)).cumsum()
    # pick the segment with the most rows
    counts = seg_id.value_counts()
    keep_seg = counts.index[0]
    return d.loc[seg_id == keep_seg].reset_index(drop=True)

def _first_entry_time_in_bbox(
    d: pd.DataFrame,
    bbox: tuple[float, float, float, float],
) -> pd.Timestamp | None:
    lon_min, lon_max, lat_min, lat_max = bbox
    inside = d["lon"].between(lon_min, lon_max) & d["lat"].between(lat_min, lat_max)
    if not inside.any():
        return None
    return d.loc[inside, "time_utc"].iloc[0]


def filter_to_first_med_entry(
    d: pd.DataFrame,
    *,
    bbox: tuple[float, float, float, float] = BBOX,
) -> pd.DataFrame:
    """
    Keep data only from the first time the platform enters the valid box bbox.
    """
    if d.empty:
        return d
    d = d.sort_values("time_utc").copy()
    t0 = _first_entry_time_in_bbox(d, bbox=bbox)
    if t0 is None:
        # No Med entry found -> return empty (explicit)
        return d.iloc[0:0].copy()
    return d[d["time_utc"] >= t0].reset_index(drop=True)

# -----------------------------------------------------------------------------
# Extensible “source” concept (for future biological variables)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtraSource:
    """
    Defines an extra dataset to merge (e.g., oxygen, chlorophyll, fluorescence).

    - name: identifier (used for lookup in paths.extra_db_dirs)
    - base_name: file prefix (e.g., 'oxygen' -> oxygen_<pid>.csv)
    - tolerance: asof tolerance when attaching to the base timeline
    - suffix: suffix to append to colliding columns
    - keep_cols: optional list of columns to keep from this dataset
    """
    name: str
    base_name: str
    tolerance: str = "30min"
    suffix: str = ""
    keep_cols: list[str] | None = None


# -----------------------------------------------------------------------------
# Master builder
# -----------------------------------------------------------------------------
def build_master_for_platform(
    platform_id: str,
    paths: MasterPaths,
    *,
    target_time: str = "drifter",
    wind_tolerance: str = "30min",
    mat_tolerance: str = "30min",
    hourly_freq: str = "1H",
    extras: list[ExtraSource] | None = None,
    # NEW:
    apply_bbox_filter: bool = True,
    apply_segment_filter: bool = False,
    segment_max_gap: str = "7D",
) -> pd.DataFrame:
    """
    Build a per-platform master table by merging canonical per-platform DBs.

    Rules (current spec):
    - from drifter: lat, lon, sst_c, slp_mb
    - from wind: all wind-related columns
    - from mat: u, v, vorticity, strain
    - keep platform_id + time_utc always
    - extras: additional datasets (e.g. oxygen) can be merged as-of with a tolerance

    target_time:
    - "drifter": base timeline is drifter timestamps; attach wind + mat (+ extras)
    - "hourly": base timeline is hourly; attach mat, drifter, wind (+ extras)
    """
    pid = str(platform_id)

    # --- read per-platform canonical DBs
    d = _read_per_platform(paths.drifter_db_dir, "drifter", pid)
    w = _read_per_platform(paths.wind_db_dir, "wind", pid)
    m = _read_per_platform(paths.mat_db_dir, "mat_timeseries", pid)

    d = _ensure_keys(d, pid)
    w = _ensure_keys(w, pid)
    m = _ensure_keys(m, pid)

    # --- select only columns we want from each source
    # Drifter: keep core + known optional + (new) lagrangian kinematics.
    # Note: _select_columns is non-strict, so missing optional columns are fine.
    drifter_keep = [
        # keys
        "platform_id",
        "time_utc",
        # core trajectory
        "lat",
        "lon",
        # common phys/met
        "sst_c",
        "slp_mb",
        # "battery_v",
        # "drogue_counts",
        # optional extended schema (if present in some chunks)
        "salinity_psu",
        # "sst_sbe_c",
        # "wind_speed_ms",
        # "wind_dir_deg",
        # NEW: lagrangian kinematics + rotation index
        "u_lag_ms",
        "v_lag_ms",
        "ax_lag_ms2",
        "ay_lag_ms2",
        "rotation_index",
    ]
    d = _select_columns(d, drifter_keep)
    # --- optional geographic + segment filters
    if apply_bbox_filter:
        d = filter_to_first_med_entry(d)

    if apply_segment_filter:
        d = filter_largest_contiguous_segment(d, max_gap=segment_max_gap)

    # wind: keep all except duplicates are handled by merge helper
    # but ensure at least keys
    if "platform_id" not in w.columns or "time_utc" not in w.columns:
        raise ValueError("Wind DB must include platform_id and time_utc")
    # mat: keep “rest”
    m = _select_columns(m, ["platform_id", "time_utc", "u", "v", "vorticity", "strain"])

    if d.empty:
        raise ValueError(f"No drifter data found for platform_id={pid} in {paths.drifter_db_dir}")
    if m.empty:
        raise ValueError(f"No MAT timeseries found for platform_id={pid} in {paths.mat_db_dir}")

    if target_time not in ("drifter", "hourly"):
        raise ValueError("target_time must be 'drifter' or 'hourly'")

    # --- build base timeline
    if target_time == "drifter":
        base = d.sort_values("time_utc").reset_index(drop=True)

        # wind onto drifter
        if not w.empty:
            base = _merge_asof_single_platform(base, w, tolerance=wind_tolerance, suffix="_wind")

        # mat onto drifter
        base = _merge_asof_single_platform(base, m, tolerance=mat_tolerance, suffix="_mat")

    else:
        # hourly base timeline driven by MAT span (consistent with its native resolution)
        tmin = m["time_utc"].min()
        tmax = m["time_utc"].max()
        timeline = pd.date_range(tmin.floor("H"), tmax.ceil("H"), freq=hourly_freq)

        base = pd.DataFrame({"time_utc": timeline})
        base.insert(0, "platform_id", pid)

        # attach MAT (often already hourly)
        base = _merge_asof_single_platform(base, m, tolerance=mat_tolerance, suffix="_mat")

        # attach drifter onto hourly
        base = _merge_asof_single_platform(base, d, tolerance="2H", suffix="_dr")

        # attach wind onto hourly
        if not w.empty:
            base = _merge_asof_single_platform(base, w, tolerance="2H", suffix="_wind")

    # --- extras (future biological datasets)
    if extras:
        if not paths.extra_db_dirs:
            raise ValueError("extras were provided but paths.extra_db_dirs is None")

        for src in extras:
            if src.name not in paths.extra_db_dirs:
                raise ValueError(f"extra_db_dirs has no entry for '{src.name}'")

            extra_df = _read_per_platform(paths.extra_db_dirs[src.name], src.base_name, pid)
            extra_df = _ensure_keys(extra_df, pid)

            if src.keep_cols:
                extra_df = _select_columns(extra_df, ["platform_id", "time_utc", *src.keep_cols])

            base = _merge_asof_single_platform(
                base,
                extra_df,
                tolerance=src.tolerance,
                suffix=src.suffix or f"_{src.name}",
            )

    return base.reset_index(drop=True)
