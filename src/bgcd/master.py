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
# QC filter helpers (remove "out of water / pre-deployment" points)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class QCParams:
    sst_min: float = 5.0
    sst_max: float = 35.0
    sal_min: float = 20.0
    sal_max: float = 45.0
    speed_max_ms: float = 3.0  # generous upper bound for SVP in m/s

def filter_drop_bad_sst_rows(
    d: pd.DataFrame,
    *,
    sst_col: str = "sst_c",
    sst_min: float = 1.0,
    sst_max: float = 40.0,
) -> pd.DataFrame:
    """
    Drop rows where SST is clearly non-physical / placeholder.
    Keeps rows if sst_col is missing.

    Rules:
      - sst must be finite
      - sst in [sst_min, sst_max]
      - sst not in common placeholders (0, -5)
    """
    if d.empty or sst_col not in d.columns:
        return d

    sst = pd.to_numeric(d[sst_col], errors="coerce")
    ok = sst.notna() & (sst >= sst_min) & (sst <= sst_max) & (~sst.isin([0.0, -5.0]))

    return d.loc[ok].copy()

def filter_trim_to_first_qc_pass(
    d: pd.DataFrame,
    *,
    qc: QCParams = QCParams(),
) -> pd.DataFrame:
    """
    Trim the initial part of the track until the first row that passes a basic
    physical QC (useful to remove pre-deployment / out-of-water points).

    QC rules (applied only if columns exist):
      - lat/lon must be finite
      - sst_c in [sst_min, sst_max] and not zero-like
      - salinity_psu in [sal_min, sal_max] and not zero-like
      - speed = sqrt(u_lag_ms^2 + v_lag_ms^2) <= speed_max_ms (if available)

    Returns data from the first QC-passing row onward.
    If no row passes QC -> returns empty.
    """
    if d.empty:
        return d

    d = d.sort_values("time_utc").copy()

    ok = d["lat"].notna() & d["lon"].notna()

    if "sst_c" in d.columns:
        sst = d["sst_c"]
        ok &= sst.notna() & (sst >= qc.sst_min) & (sst <= qc.sst_max) & (sst != 0.0)

    if "salinity_psu" in d.columns:
        sal = d["salinity_psu"]

        # Treat placeholders as missing
        sal_clean = sal.where(~sal.isin([0.0, 15.0]))

        # Use salinity as QC constraint ONLY if we have at least one plausible value
        good_sal = sal_clean.notna() & (sal_clean >= qc.sal_min) & (sal_clean <= qc.sal_max)

        if good_sal.any():
            ok &= good_sal
        # else: ignore salinity entirely for QC (sensor not available / placeholder)

    if "u_lag_ms" in d.columns and "v_lag_ms" in d.columns:
        u = d["u_lag_ms"]
        v = d["v_lag_ms"]
        spd = (u.astype("float64") ** 2 + v.astype("float64") ** 2) ** 0.5
        ok &= spd.notna() & (spd <= qc.speed_max_ms)

    if not ok.any():
        return d.iloc[0:0].copy()

    t0 = d.loc[ok, "time_utc"].iloc[0]
    return d[d["time_utc"] >= t0].reset_index(drop=True)
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
    tag: str | None = None,
    with_dt: bool = False,
) -> pd.DataFrame:
    """
    Nearest-time merge_asof for a single platform (platform_id already filtered).
    Keeps all left rows.

    If with_dt=True, keeps matched right timestamps and adds |Δt| diagnostics:
      - time_utc_<tag>
      - dt_<tag>_min
    """
    l = left.sort_values("time_utc").copy()
    r = right.sort_values("time_utc").copy()

    # If diagnostics requested, keep right time under a different column name
    if with_dt:
        tcol = f"time_utc_{tag}" if tag else "time_utc_right"
        r = r.rename(columns={"time_utc": tcol})
        left_on = "time_utc"
        right_on = tcol
    else:
        left_on = right_on = "time_utc"

    # avoid duplicate column names (except keys + time columns)
    protected = {"platform_id", left_on, right_on}
    r_cols = [c for c in r.columns if c not in protected]

    ren = {c: f"{c}{suffix}" for c in r_cols if c in l.columns}
    if ren:
        r = r.rename(columns=ren)

    merged = pd.merge_asof(
        l,
        r.drop(columns=["platform_id"]),
        left_on=left_on,
        right_on=right_on,
        direction=direction,
        tolerance=pd.Timedelta(tolerance),
    )

    if with_dt:
        dt_col = f"dt_{tag}_min" if tag else "dt_right_min"
        merged[dt_col] = (merged[right_on] - merged["time_utc"]).abs() / pd.Timedelta("1min")

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
    apply_bbox_filter: bool = True,
    apply_segment_filter: bool = False,
    segment_max_gap: str = "7D",
    with_dt: bool = False,
    # QC:
    apply_qc_trim: bool = True,
    qc: QCParams = QCParams(),
    apply_sst_qc: bool = True,
) -> pd.DataFrame:
    """
    Build a per-platform master table by merging canonical per-platform DBs.

    Rules (current spec):
    - from drifter: lat, lon, sst_c, slp_mb (+ optional extended schema)
    - from wind: all wind-related columns
    - from mat: u, v, vorticity, strain, okubo_weiss
    - keep platform_id + time_utc always
    - extras: additional datasets (e.g. oxygen) can be merged as-of with a tolerance

    target_time:
    - "drifter": base timeline is drifter timestamps; attach wind + mat (+ extras)
    - "hourly": base timeline is hourly; attach mat, drifter, wind (+ extras)

    with_dt:
    - if True, keep matched timestamps from each source and add dt_<source>_min diagnostics
      (requires _merge_asof_single_platform to support tag + with_dt)
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
        # optional extended schema (if present in some chunks)
        "salinity_psu",
        # lagrangian kinematics + rotation index (if present)
        "u_lag_ms",
        "v_lag_ms",
        "ax_lag_ms2",
        "ay_lag_ms2",
        "rotation_index",
        "curvature_m1",
        "curvature_signed_m1",
    ]
    d = _select_columns(d, drifter_keep)

    # --- optional geographic + segment filters
    if apply_bbox_filter:
        d = filter_to_first_med_entry(d)

    if apply_segment_filter:
        d = filter_largest_contiguous_segment(d, max_gap=segment_max_gap)

    # --- NEW: row-wise SST QC cleanup (uses qc.sst_min/max)
    if apply_sst_qc:
        d = filter_drop_bad_sst_rows(d, sst_min=qc.sst_min, sst_max=qc.sst_max)

    # --- optional trim (uses broader QC rules)
    if apply_qc_trim:
        d = filter_trim_to_first_qc_pass(d, qc=qc)

    # wind: keep all columns (only ensure keys)
    if "platform_id" not in w.columns or "time_utc" not in w.columns:
        raise ValueError("Wind DB must include platform_id and time_utc")

    # mat: keep a subset
    m = _select_columns(m, ["platform_id", "time_utc", "u", "v", "vorticity", "strain", "okubo_weiss"])

    if d.empty:
        raise ValueError(f"No drifter data found for platform_id={pid} in {paths.drifter_db_dir}")
    if m.empty:
        raise ValueError(f"No MAT timeseries found for platform_id={pid} in {paths.mat_db_dir}")

    if target_time not in ("drifter", "hourly"):
        raise ValueError("target_time must be 'drifter' or 'hourly'")

    # --- build base timeline
    if target_time == "drifter":
        base = d.sort_values("time_utc").reset_index(drop=True)

        # wind onto drifter timeline
        if not w.empty:
            base = _merge_asof_single_platform(
                base,
                w,
                tolerance=wind_tolerance,
                suffix="_wind",
                tag="wind",
                with_dt=with_dt,
            )

        # mat onto drifter timeline
        base = _merge_asof_single_platform(
            base,
            m,
            tolerance=mat_tolerance,
            suffix="_mat",
            tag="mat",
            with_dt=with_dt,
        )

    else:
        # hourly base timeline driven by MAT span (consistent with its native resolution)
        tmin = m["time_utc"].min()
        tmax = m["time_utc"].max()
        timeline = pd.date_range(tmin.floor("H"), tmax.ceil("H"), freq=hourly_freq)

        base = pd.DataFrame({"time_utc": timeline})
        base.insert(0, "platform_id", pid)

        # attach MAT (often already hourly)
        base = _merge_asof_single_platform(
            base,
            m,
            tolerance=mat_tolerance,
            suffix="_mat",
            tag="mat",
            with_dt=with_dt,
        )

        # attach drifter onto hourly
        base = _merge_asof_single_platform(
            base,
            d,
            tolerance="2H",
            suffix="_dr",
            tag="drifter",
            with_dt=with_dt,
        )

        # attach wind onto hourly
        if not w.empty:
            base = _merge_asof_single_platform(
                base,
                w,
                tolerance="2H",
                suffix="_wind",
                tag="wind",
                with_dt=with_dt,
            )

    # --- extras (e.g. oxygen)
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
                tag=src.name,          # <-- oxygen -> time_utc_oxygen + dt_oxygen_min
                with_dt=with_dt,
            )

    return base.reset_index(drop=True)
