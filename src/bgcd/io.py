from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd


def _preclean_lines(path: str | Path) -> str:
    """
    Minimal pre-clean for CSV exported with junk HTML break tokens.
    - removes empty lines
    - removes '</br>' tokens
    - keeps only the first header occurrence (if repeated)
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    cleaned = []
    header_seen = False

    for ln in lines:
        ln = ln.replace("</br>", "").strip()
        if not ln:
            continue

        # Detect header repetition for these datasets
        if ln.startswith("Platform-ID"):
            if header_seen:
                continue
            header_seen = True

        cleaned.append(ln)

    return "\n".join(cleaned)


def read_drifter_csv(path: str | Path) -> pd.DataFrame:
    """
    Read drifter_data.csv and normalize to canonical columns:
    platform_id, time_utc, lat, lon, sst_c, slp_mb, battery_v, drogue_counts
    """
    csv_text = _preclean_lines(path)
    df = pd.read_csv(
        StringIO(csv_text),
        skipinitialspace=True,
        engine="python",
        on_bad_lines="skip",
    )

    # Drop empty/unnamed columns from trailing commas
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")].copy()

    rename = {
        "Platform-ID": "platform_id",
        "Timestamp(UTC)": "time_utc",
        "GPS-Latitude(deg)": "lat",
        "GPS-Longitude(deg)": "lon",
        "SST(degC)": "sst_c",
        "SLP(mB)": "slp_mb",
        "Battery(volts)": "battery_v",
        "Drogue (cnts)": "drogue_counts",
    }
    df.rename(columns=rename, inplace=True)

    df["platform_id"] = df["platform_id"].astype(str).str.strip()
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    for c in ["lat", "lon", "sst_c", "slp_mb", "battery_v", "drogue_counts"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["platform_id", "time_utc", "lat", "lon"])
    df = df.sort_values(["platform_id", "time_utc"]).reset_index(drop=True)
    return df


def read_wind_csv(path: str | Path) -> pd.DataFrame:
    """
    Read wind_data.csv and normalize to canonical columns:
    platform_id, time_utc, wspd*, wdir*, samples
    Also fixes the known typo Widr_skewness -> Wdir_skewness.
    """
    csv_text = _preclean_lines(path)
    df = pd.read_csv(
        StringIO(csv_text),
        skipinitialspace=True,
        engine="python",
        on_bad_lines="skip",
    )

    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")].copy()

    # Fix typo in provided header
    if "Widr_skewness" in df.columns and "Wdir_skewness" not in df.columns:
        df.rename(columns={"Widr_skewness": "Wdir_skewness"}, inplace=True)

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
    df.rename(columns=rename, inplace=True)

    df["platform_id"] = df["platform_id"].astype(str).str.strip()
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    for c in df.columns:
        if c in ("platform_id", "time_utc"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "samples" in df.columns:
        df["samples"] = df["samples"].round().astype("Int64")

    df = df.dropna(subset=["platform_id", "time_utc"])
    df = df.sort_values(["platform_id", "time_utc"]).reset_index(drop=True)
    return df


def merge_drifter_wind(
    drifter: pd.DataFrame,
    wind: pd.DataFrame,
    tolerance: str = "30min",
) -> pd.DataFrame:
    """
    Nearest-time merge of wind onto drifter observations, per platform_id.
    Keeps all drifter rows.
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
