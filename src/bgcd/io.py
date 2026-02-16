from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _preclean_lines(path: str | Path) -> str:
    """
    Minimal pre-clean for CSV exported with junk HTML break tokens.
    - removes empty lines
    - removes '</br>' tokens

    Note:
    This does NOT solve multi-header / multi-schema CSV containers.
    For that, use bgcd.raw_split + per-platform outputs, then canonicalize_*.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    cleaned = []
    for ln in lines:
        ln = ln.replace("</br>", "").strip()
        if ln:
            cleaned.append(ln)

    return "\n".join(cleaned)


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains(r"^Unnamed")].copy()


def _coerce_numeric(df: pd.DataFrame, exclude: Iterable[str]) -> pd.DataFrame:
    for c in df.columns:
        if c in exclude:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -----------------------------------------------------------------------------
# Canonicalization (recommended path)
# -----------------------------------------------------------------------------
def canonicalize_drifter_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize a drifter dataframe (typically coming from a per-platform RAW CSV).

    Required canonical columns:
      - platform_id
      - time_utc
      - lat
      - lon

    Optional if present:
      - sst_c, slp_mb, battery_v, drogue_counts
      - salinity_psu, sst_sbe_c, wind_speed_ms, wind_dir_deg
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename = {
        "Platform-ID": "platform_id",
        "Timestamp(UTC)": "time_utc",
        "GPS-Latitude(deg)": "lat",
        "GPS-Longitude(deg)": "lon",
        "SST(degC)": "sst_c",
        "SLP(mB)": "slp_mb",
        "Battery(volts)": "battery_v",
        "Drogue (cnts)": "drogue_counts",
        # extended schema
        "Salinity(PSU)": "salinity_psu",
        "sstSBE (degC)": "sst_sbe_c",
        "Wind-Speed (m/s)": "wind_speed_ms",
        "Wind-Direction (deg)": "wind_dir_deg",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    required = ["platform_id", "time_utc", "lat", "lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    df["platform_id"] = df["platform_id"].astype(str).str.strip()
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    df = _coerce_numeric(df, exclude=("platform_id", "time_utc"))
    df = df.dropna(subset=["platform_id", "time_utc", "lat", "lon"])
    df = df.sort_values(["platform_id", "time_utc"]).reset_index(drop=True)
    return df


def canonicalize_wind_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize a wind dataframe (typically coming from a per-platform RAW CSV).

    Required canonical columns:
      - platform_id
      - time_utc

    Wind columns if present:
      - wspd, wspd_mean, wspd_min, wspd_max, wspd_std, wspd_skewness, wspd_kurtosis
      - wdir, wdir_mean, wdir_min, wdir_max, wdir_std, wdir_skewness, wdir_kurtosis
      - samples
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Drop garbage columns (e.g. ';;;;;;;') sometimes present in exports
    df = df.loc[:, ~df.columns.str.fullmatch(r";+")]

    # Fix known typo if present
    if "Widr_skewness" in df.columns and "Wdir_skewness" not in df.columns:
        df = df.rename(columns={"Widr_skewness": "Wdir_skewness"})

    rename = {
        "Platform-ID": "platform_id",
        "GPS-Timestamp(utc)": "time_utc",
        "Wspd(m/s)": "wspd",
        "Wspd_mean(m/s)": "wspd_mean",
        "Wspd_min(m/s)": "wspd_min",
        "Wspd_max(m/s)": "wspd_max",
        "Wspd_std(m/s)": "wspd_std",
        "Wspd_skewness": "wspd_skewness",
        "Wspd_kurtosis": "wspd_kurtosis",
        "Wdir(deg)": "wdir",
        "Wdir_mean(deg)": "wdir_mean",
        "Wdir_min(deg)": "wdir_min",
        "Wdir_max(deg)": "wdir_max",
        "Wdir_std(deg)": "wdir_std",
        "Wdir_skewness": "wdir_skewness",
        "Wdir_kurtosis": "wdir_kurtosis",
        "Samples": "samples",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    required = ["platform_id", "time_utc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    df["platform_id"] = df["platform_id"].astype(str).str.strip()
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    df = _coerce_numeric(df, exclude=("platform_id", "time_utc"))

    if "samples" in df.columns:
        df["samples"] = df["samples"].round().astype("Int64")

    df = df.dropna(subset=["platform_id", "time_utc"])
    df = df.sort_values(["platform_id", "time_utc"]).reset_index(drop=True)
    return df


def read_drifter_platform_file(path: str | Path) -> pd.DataFrame:
    """
    Read a per-platform RAW drifter CSV (e.g., db_drifter_raw/drifter_raw_<id>.csv)
    and return a canonicalized DataFrame.
    """
    df = pd.read_csv(path)
    df = _drop_unnamed(df)
    return canonicalize_drifter_df(df)


def read_wind_platform_file(path: str | Path) -> pd.DataFrame:
    """
    Read a per-platform RAW wind CSV (e.g., db_wind_raw/wind_raw_<id>.csv)
    and return a canonicalized DataFrame.
    """
    df = pd.read_csv(path)
    df = _drop_unnamed(df)
    return canonicalize_wind_df(df)


# -----------------------------------------------------------------------------
# Legacy: read whole raw CSV (best-effort, NOT recommended for multi-header files)
# -----------------------------------------------------------------------------
def read_drifter_csv(path: str | Path) -> pd.DataFrame:
    """
    Best-effort read of the *raw* drifter_data.csv.

    WARNING:
    If the file contains multiple header blocks with different schemas,
    this function may skip or mis-parse parts of the file.

    Recommended workflow:
    - split into per-platform raw files (bgcd.raw_split)
    - then read per-platform via read_drifter_platform_file
    """
    csv_text = _preclean_lines(path)
    df = pd.read_csv(StringIO(csv_text), skipinitialspace=True, engine="python", on_bad_lines="skip")
    df = _drop_unnamed(df)
    return canonicalize_drifter_df(df)


def read_wind_csv(path: str | Path) -> pd.DataFrame:
    """
    Best-effort read of the *raw* wind_data.csv.

    WARNING:
    If the file contains multiple header blocks with different schemas,
    this function may skip or mis-parse parts of the file.

    Recommended workflow:
    - split into per-platform raw files (bgcd.raw_split)
    - then read per-platform via read_wind_platform_file
    """
    csv_text = _preclean_lines(path)
    df = pd.read_csv(StringIO(csv_text), skipinitialspace=True, engine="python", on_bad_lines="skip")
    df = _drop_unnamed(df)
    return canonicalize_wind_df(df)


# -----------------------------------------------------------------------------
# Merge utilities
# -----------------------------------------------------------------------------
def merge_drifter_wind(
    drifter: pd.DataFrame,
    wind: pd.DataFrame,
    tolerance: str = "30min",
) -> pd.DataFrame:
    """
    Nearest-time merge of wind onto drifter observations, per platform_id.
    Keeps all drifter rows.

    Expects canonical columns:
      drifter: platform_id, time_utc
      wind:    platform_id, time_utc
    """
    d = drifter.sort_values(["platform_id", "time_utc"]).copy()
    w = wind.sort_values(["platform_id", "time_utc"]).copy()

    out = []
    for pid, dpid in d.groupby("platform_id", sort=False):
        wpid = w[w["platform_id"] == pid]
        if wpid.empty:
            out.append(dpid)
            continue

        merged = pd.merge_asof(
            dpid,
            wpid.drop(columns=["platform_id"]),
            on="time_utc",
            direction="nearest",
            tolerance=pd.Timedelta(tolerance),
        )
        out.append(merged)

    return pd.concat(out, ignore_index=True)




# -----------------------------------------------------------------------------
# Kinematics
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Kinematics
# -----------------------------------------------------------------------------
def add_lagrangian_kinematics(
    df: pd.DataFrame,
    *,
    group_col: str = "platform_id",
    time_col: str = "time_utc",
    lat_col: str = "lat",
    lon_col: str = "lon",
    earth_radius_m: float = 6371000.0,
    # NEW: threshold to avoid curvature blow-ups at near-zero speed
    min_speed_ms: float = 1e-3,
) -> pd.DataFrame:
    """
    Add Lagrangian velocity/acceleration components (east/north), rotation index,
    and trajectory curvature.

    Output columns:
      - u_lag_ms, v_lag_ms
      - ax_lag_ms2, ay_lag_ms2
      - rotation_index  (sin(angle between velocity unit vector and acceleration unit vector))
      - curvature_m1            (|v x a| / |v|^3)  [1/m]
      - curvature_signed_m1     ((v x a) / |v|^3)  [1/m]

    Notes
    -----
    Curvature definition (2D):
      cross = u*ay - v*ax
      kappa_abs = |cross| / (u^2+v^2)^(3/2)
      kappa_sgn =  cross  / (u^2+v^2)^(3/2)

    Curvature becomes unstable when speed -> 0. We set curvature to NaN when
    speed < min_speed_ms.
    """
    out = df.copy()

    # ensure ordering
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)

    # allocate columns
    kin_cols = [
        "u_lag_ms", "v_lag_ms",
        "ax_lag_ms2", "ay_lag_ms2",
        "rotation_index",
        "curvature_m1", "curvature_signed_m1",
    ]
    for c in kin_cols:
        out[c] = np.nan

    def _finite_diff(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        x: position component (m) or velocity component (m/s)
        t: time in seconds (float)
        returns dx/dt with central differences (ends: forward/backward)
        """
        n = len(x)
        y = np.full(n, np.nan, dtype=float)
        if n < 2:
            return y

        # forward/backward
        dt0 = t[1] - t[0]
        dtn = t[-1] - t[-2]
        if dt0 != 0:
            y[0] = (x[1] - x[0]) / dt0
        if dtn != 0:
            y[-1] = (x[-1] - x[-2]) / dtn

        if n >= 3:
            dt = t[2:] - t[:-2]
            num = x[2:] - x[:-2]
            mask = dt != 0
            y[1:-1][mask] = num[mask] / dt[mask]

        return y

    for pid, g in out.groupby(group_col, sort=False):
        if g.empty:
            continue

        # time in seconds
        t = pd.to_datetime(g[time_col], errors="coerce")
        t_s = (t.astype("int64") / 1e9).to_numpy(dtype=float)  # seconds since epoch

        lat = pd.to_numeric(g[lat_col], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(g[lon_col], errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(t_s) & np.isfinite(lat) & np.isfinite(lon)
        if ok.sum() < 2:
            continue

        idx = g.index.to_numpy()
        idx_ok = idx[ok]
        t_ok = t_s[ok]
        lat_ok = np.deg2rad(lat[ok])
        lon_ok = np.deg2rad(lon[ok])

        # local dx,dy in meters
        dlat = np.diff(lat_ok, prepend=lat_ok[0])
        dlon = np.diff(lon_ok, prepend=lon_ok[0])

        dx = earth_radius_m * np.cos(lat_ok) * dlon
        dy = earth_radius_m * dlat

        x = np.cumsum(dx)  # meters
        y = np.cumsum(dy)  # meters

        u = _finite_diff(x, t_ok)   # m/s east
        v = _finite_diff(y, t_ok)   # m/s north
        ax = _finite_diff(u, t_ok)  # m/s^2 east
        ay = _finite_diff(v, t_ok)  # m/s^2 north

        # rotation index: cross product of unit vectors (v_hat x a_hat)_z
        vnorm = np.sqrt(u*u + v*v)
        anorm = np.sqrt(ax*ax + ay*ay)

        rot = np.full_like(u, np.nan, dtype=float)
        eps = 1e-12
        m_rot = (vnorm > eps) & (anorm > eps)
        uhat = u[m_rot] / vnorm[m_rot]
        vhat = v[m_rot] / vnorm[m_rot]
        axhat = ax[m_rot] / anorm[m_rot]
        ayhat = ay[m_rot] / anorm[m_rot]
        rot[m_rot] = uhat * ayhat - vhat * axhat

        # NEW: curvature (absolute and signed)
        cross = u * ay - v * ax  # (v x a)_z
        denom = (u*u + v*v) ** 1.5

        kappa_abs = np.full_like(u, np.nan, dtype=float)
        kappa_sgn = np.full_like(u, np.nan, dtype=float)

        m_k = (vnorm >= float(min_speed_ms)) & (denom > 0)
        kappa_abs[m_k] = np.abs(cross[m_k]) / denom[m_k]
        kappa_sgn[m_k] = cross[m_k] / denom[m_k]

        out.loc[idx_ok, "u_lag_ms"] = u
        out.loc[idx_ok, "v_lag_ms"] = v
        out.loc[idx_ok, "ax_lag_ms2"] = ax
        out.loc[idx_ok, "ay_lag_ms2"] = ay
        out.loc[idx_ok, "rotation_index"] = rot
        out.loc[idx_ok, "curvature_m1"] = kappa_abs
        out.loc[idx_ok, "curvature_signed_m1"] = kappa_sgn

    return out
