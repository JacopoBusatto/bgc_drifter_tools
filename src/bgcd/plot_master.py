#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import matplotlib as mpl

fontSize=22
mpl.rcParams.update({
    "font.size": fontSize,          # dimensione base
    "axes.titlesize": fontSize,     # titolo del plot
    "axes.labelsize": fontSize,     # etichette assi
    "xtick.labelsize": fontSize,    # numeri asse X
    "ytick.labelsize": fontSize,    # numeri asse Y
    "legend.fontsize": fontSize,    # legenda
})

# ----------------------------
# Loading + light QC
# ----------------------------
def read_master(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # time
    if "time_utc" not in df.columns:
        raise ValueError("Expected column 'time_utc' in MASTER.")
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    # platform
    if "platform_id" in df.columns:
        df["platform_id"] = df["platform_id"].astype(str).str.strip()

    # numeric
    for c in df.columns:
        if c in ("platform_id", "time_utc"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    return df


def apply_basic_masks(
    df: pd.DataFrame,
    *,
    sst_col: str = "sst_c",
    drop_sst_values: tuple[float, ...] = (-5.0,),
    treat_zero_sst_as_nan: bool = True,
    vort_col: str = "vorticity",
) -> pd.DataFrame:
    """
    Mask common fill values.
    - SST sometimes uses -5.0 and/or 0.0 as fill.
    - Some diagnostic variables may use ~999 as fill.
    """
    out = df.copy()

    if sst_col in out.columns:
        if drop_sst_values:
            out.loc[out[sst_col].isin(drop_sst_values), sst_col] = np.nan
        if treat_zero_sst_as_nan:
            out.loc[out[sst_col] == 0.0, sst_col] = np.nan

    # common fill for winds / diagnostics in some pipelines
    for c in out.columns:
        if c.endswith(("u", "v")) or c in (vort_col, "strain"):
            out.loc[out[c].abs() >= 900, c] = np.nan

    return out


# ----------------------------
# Helpers
# ----------------------------
def clip_series_quantile(series: pd.Series, qlow: float = 0.01, qhigh: float = 0.99) -> pd.Series:
    """
    Clip a series to given quantile range.
    Useful for removing extreme numerical outliers (e.g. curvature spikes).
    """
    s = series.copy()
    if s.notna().sum() < 5:
        return s

    lo = s.quantile(qlow)
    hi = s.quantile(qhigh)
    return s.clip(lower=lo, upper=hi)

def compute_hourly_climatology_mean(
    t: pd.Series,
    y: pd.Series,
    *,
    min_samples_per_hour: int = 3,
) -> pd.Series:
    """
    Compute mean value for each hour-of-day (0..23) across the whole record.
    Returns a Series indexed by hour (0..23) with NaN where not enough samples.
    """
    tt = pd.to_datetime(t, errors="coerce")
    tmp = pd.DataFrame({"t": tt, "y": y}).dropna(subset=["t", "y"])
    if tmp.empty:
        return pd.Series(index=np.arange(24), dtype=float)

    hours = tmp["t"].dt.hour
    g = tmp["y"].groupby(hours)
    mean_by_hour = g.mean().reindex(np.arange(24))
    count_by_hour = g.size().reindex(np.arange(24)).fillna(0).astype(int)

    mean_by_hour[count_by_hour < int(min_samples_per_hour)] = np.nan
    return mean_by_hour


def hourly_anomaly(
    t: pd.Series,
    y: pd.Series,
    *,
    min_samples_per_hour: int = 3,
) -> pd.Series:
    """
    y(t) - mean(y | hour-of-day), computed over the whole time series.
    """
    tt = pd.to_datetime(t, errors="coerce")
    clim = compute_hourly_climatology_mean(tt, y, min_samples_per_hour=min_samples_per_hour)
    h = tt.dt.hour
    return y - h.map(clim)


@dataclass(frozen=True)
class AnomalyOpts:
    enabled: bool = False
    min_samples_per_hour: int = 3

def wind_dirspeed_to_uv(wdir_deg: np.ndarray, wspd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert meteorological direction (deg FROM which wind is coming, clockwise from North)
    to map components (u east, v north) for an arrow pointing TOWARDS where it goes.

    u = -wspd * sin(theta)
    v = -wspd * cos(theta)
    """
    theta = np.deg2rad(wdir_deg)
    u = -wspd * np.sin(theta)
    v = -wspd * np.cos(theta)
    return u, v


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_time_markers(df: pd.DataFrame, every_days: int = 7) -> pd.DataFrame:
    """
    Compute numbered markers every N days using the first sample in each resample bin.
    Returns a dataframe with columns: ['marker_id', 'time_utc', 'lon', 'lat'].
    """
    if every_days <= 0 or "time_utc" not in df.columns:
        return pd.DataFrame(columns=["marker_id", "time_utc", "lon", "lat"])

    mk = (
        df[["time_utc", "lon", "lat"]]
        .dropna(subset=["time_utc", "lon", "lat"])
        .sort_values("time_utc")
        .set_index("time_utc")
        .resample(f"{int(every_days)}D")
        .first()
        .dropna(subset=["lon", "lat"])
        .reset_index()
    )

    if mk.empty:
        return pd.DataFrame(columns=["marker_id", "time_utc", "lon", "lat"])

    mk.insert(0, "marker_id", np.arange(1, len(mk) + 1))
    return mk

def rolling_time_mean(series: pd.Series, time: pd.Series, window: str) -> pd.Series:
    """
    Time-based rolling mean using a pandas offset window (e.g. '6H', '30min', '1D').
    Works with irregular sampling.
    """
    s = pd.Series(series.to_numpy(), index=pd.DatetimeIndex(time))
    s = s.sort_index()
    return s.rolling(window=window, min_periods=1).mean().reindex(pd.DatetimeIndex(time)).to_numpy()



# ----------------------------
# Plot: single-variable time series (one PNG per variable)
# ----------------------------
def plot_single_ts(
    df: pd.DataFrame,
    outpath: Path,
    *,
    title: str,
    col: str,
    ylabel: str,
    time_col: str = "time_utc",
    smooth_window: str | None = None,
    markers: pd.DataFrame | None = None,
    anomaly: AnomalyOpts = AnomalyOpts(),
    logy: bool = False,
) -> None:
    """
    Plot one variable vs time.
    If smooth_window is provided (e.g. '6H'), applies a time-based rolling mean.
    """
    if col not in df.columns:
        return

    t = df[time_col]
    y = df[col]

    # Hard-coded clipping for curvature variables
    if col in ("curvature_m1", "curvature_signed_m1"):
        y = clip_series_quantile(y, 0.01, 0.99)
    if col in ("DO2_c", "oxy_comp_mgL_c"):
        y = clip_series_quantile(y, 0.01, 0.99)

    # keep original (after clipping but before smoothing)
    y_raw = y.copy()

    # compute anomaly on raw signal
    y_anom = None
    if anomaly.enabled:
        y_anom = hourly_anomaly(
            t,
            y_raw,
            min_samples_per_hour=anomaly.min_samples_per_hour,
        )

    # now apply smoothing (if requested)
    if smooth_window:
        y_sm = rolling_time_mean(y_raw, t, smooth_window)
        y = pd.Series(y_sm, index=y.index)

        if anomaly.enabled and y_anom is not None:
            y_anom_sm = rolling_time_mean(y_anom, t, smooth_window)
            y_anom = pd.Series(y_anom_sm, index=y.index)
    else:
        y = y_raw

    fig = plt.figure(figsize=(12.5, 4.8))
    ax = plt.gca()
    ax.grid(True, which="both", alpha=0.35)

    # --- Main signal ---
    main_color = ax._get_lines.get_next_color()
    line_main, = ax.plot(
        t,
        y,
        linewidth=1.3,
        color=main_color,
        label="Signal",
    )

    legend_handles = [line_main]
    legend_labels = ["Signal"]

    ax2 = None

    # --- Hourly anomaly on right axis ---
    if anomaly.enabled and y_anom is not None and pd.Series(y_anom).notna().any():
        # IMPORTANT: anomaly + logy non vanno d'accordo (anomaly può essere negativa)
        ax2 = ax.twinx()
        anom_color = ax._get_lines.get_next_color()

        line_anom, = ax2.plot(
            t,
            y_anom,
            linewidth=1.3,
            linestyle="--",
            color=anom_color,
            label="Hourly anomaly",
        )

        ax2.axhline(0.0, linewidth=0.9, alpha=0.35)
        ax2.set_ylabel(f"{ylabel} anomaly")

        legend_handles.append(line_anom)
        legend_labels.append("Hourly anomaly")

    # --- Legend (single, clean) ---
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        fontsize=fontSize,
        frameon=True,
        framealpha=0.85,
    )

    # ------------------------------------------------------------------
    # Vertical marker lines (every 15 days)
    # ------------------------------------------------------------------
    if t.notna().any():
        tmin = pd.to_datetime(t.min())
        tmax = pd.to_datetime(t.max())

        # primo marker allineato al giorno intero
        start = tmin.normalize()

        marker_times = pd.date_range(start=start, end=tmax, freq="15D")

        for i, mt in enumerate(marker_times, start=1):
            ax.axvline(mt, linewidth=0.9, alpha=0.25)

            ax.annotate(
                str(i),
                xy=(mt, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -2),
                textcoords="offset points",
                fontsize=fontSize,
                ha="left",
                va="top",
                alpha=0.75,
            )

        # Allinea anche le xticks ogni 15 giorni
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(ylabel)

    if logy:
        # usa la serie effettivamente plottata (y) per stimare bottom, non df[col]
        y_series = pd.Series(y)
        y_positive = y_series[y_series > 0]
        if not y_positive.empty:
            ax.set_ylim(bottom=float(y_positive.min()) * 0.5)
        ax.set_yscale("log")

    # formatter senza anno
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    if ax2 is not None:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    # piccola rotazione per evitare collisioni se ci sono tanti tick
    # plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_xticklabels(), ha="right")

    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_time_series_singles(
    df: pd.DataFrame,
    outdir: Path,
    *,
    title_prefix: str = "",
    smooth_all: str | None = None,
    rot_smooth: str | None = None,
    markers: pd.DataFrame | None = None,  # NEW
    anomaly: AnomalyOpts = AnomalyOpts(),
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    sst_col    = "sst_c"               if "sst_c" in df.columns else None
    sal_col    = "salinity_psu"        if "salinity_psu" in df.columns else None
    wspd_col   = pick_first_existing(df, ["wspd_mean", "wspd"])
    vort_col   = "vorticity"           if "vorticity" in df.columns else None
    rot_col    = "rotation_index"      if "rotation_index" in df.columns else None
    curv_col   = "curvature_m1"        if "curvature_m1" in df.columns else None
    curv_s_col = "curvature_signed_m1" if "curvature_signed_m1" in df.columns else None
    strain_col = "strain"              if "strain" in df.columns else None
    ow_col     = "okubo_weiss"         if "okubo_weiss" in df.columns else None
    do2_raw_col = "DO2_c" if "DO2_c" in df.columns else None
    do2_mgl_col = "oxy_comp_mgL_c" if "oxy_comp_mgL_c" in df.columns else None
    bbp470_col = "bbp_470_m1" if "bbp_470_m1" in df.columns else None
    bbp532_col = "bbp_532_m1" if "bbp_532_m1" in df.columns else None

    if sst_col:
        plot_single_ts(
            df,
            outdir / "ts_sst.png",
            title=f"{title_prefix}SST" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=sst_col,
            ylabel="SST (°C)",
            smooth_window=smooth_all,
            markers=markers,  # NEW
            anomaly=anomaly,
        )

    if sal_col:
        plot_single_ts(
            df,
            outdir / "ts_salinity.png",
            title=f"{title_prefix}Salinity" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=sal_col,
            ylabel="Salinity (PSU)",
            smooth_window=smooth_all,
            markers=markers,  # NEW
            anomaly=anomaly,
        )

    if wspd_col:
        plot_single_ts(
            df,
            outdir / "ts_wind_speed.png",
            title=f"{title_prefix}Wind speed" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=wspd_col,
            ylabel="Wind speed (m/s)",
            smooth_window=smooth_all,
            markers=markers,  # NEW
            anomaly=anomaly,
        )

    if vort_col:
        plot_single_ts(
            df,
            outdir / "ts_vorticity.png",
            title=f"{title_prefix}Vorticity" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=vort_col,
            ylabel="Vorticity (1/s)",
            smooth_window=smooth_all,
            markers=markers,  # NEW
            anomaly=anomaly,
        )

    if strain_col:
        plot_single_ts(
            df,
            outdir / "ts_strain.png",
            title=f"{title_prefix}Strain" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=strain_col,
            ylabel="Strain (1/s)",
            smooth_window=smooth_all,
            markers=markers,  # NEW
            anomaly=anomaly,
        )
    
    if rot_col:
        rot_window = rot_smooth or smooth_all
        suffix = f"_smooth_{rot_window}" if rot_window else ""
        plot_single_ts(
            df,
            outdir / f"ts_rotation_index{suffix}.png",
            title=f"{title_prefix}Rotation index" + (f" (smooth {rot_window})" if rot_window else ""),
            col=rot_col,
            ylabel="Rotation index (-)",
            smooth_window=rot_window,
            markers=markers,  # NEW
            anomaly=anomaly,
        )

    if curv_col:
        plot_single_ts(
            df,
            outdir / "ts_curvature.png",
            title=f"{title_prefix}Curvature" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=curv_col,
            ylabel="Curvature (m$^{-1}$)",
            smooth_window=smooth_all,
            markers=markers,  # NEW
            anomaly=anomaly,
        )

    if curv_s_col:
        plot_single_ts(
            df,
            outdir / "ts_curvature_signed.png",
            title=f"{title_prefix}Signed curvature" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=curv_s_col,
            ylabel="Signed curvature (m$^{-1}$)",
            smooth_window=smooth_all,
            markers=markers,  # NEW
            anomaly=anomaly,
        )

    if ow_col:
        plot_single_ts(
            df,
            outdir / "ts_okubo_weiss.png",
            title=f"{title_prefix}Okubo–Weiss" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=ow_col,
            ylabel="Okubo–Weiss (1/s$^2$)",
            smooth_window=smooth_all,
            markers=markers,  # NEW
            anomaly=anomaly,
        )

    if do2_raw_col:
        plot_single_ts(
            df,
            outdir / "ts_oxygen_DO2_raw.png",
            title=f"{title_prefix}DO2 (raw counts)" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=do2_raw_col,
            ylabel="DO2 (counts)",
            smooth_window=smooth_all,
            markers=markers,
            anomaly=anomaly,
        )

    if do2_mgl_col:
        plot_single_ts(
            df,
            outdir / "ts_oxygen_mgL.png",
            title=f"{title_prefix}Oxygen (mg/L)" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=do2_mgl_col,
            ylabel="Oxygen (mg/L)",
            smooth_window=smooth_all,
            markers=markers,
            anomaly=anomaly,
        )

    if bbp470_col:
        plot_single_ts(
            df,
            outdir / "ts_bbp_470.png",
            title=f"{title_prefix}BBP 470 nm" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=bbp470_col,
            ylabel="bbp (m$^{-1}$)",
            smooth_window=smooth_all,
            markers=markers,
            anomaly=anomaly,
            logy=False,
        )

    if bbp532_col:
        plot_single_ts(
            df,
            outdir / "ts_bbp_532.png",
            title=f"{title_prefix}BBP 532 nm" + (f" (smooth {smooth_all})" if smooth_all else ""),
            col=bbp532_col,
            ylabel="bbp (m$^{-1}$)",
            smooth_window=smooth_all,
            markers=markers,
            anomaly=anomaly,
            logy=False,
        )
# ----------------------------
# Plot 1: Cartopy trajectory map (PlateCarree) + vorticity colors + wind arrows
# ----------------------------
def plot_trajectory_cartopy(
    df: pd.DataFrame,
    outdir: Path,
    *,
    title_prefix: str = "",
    decimate_quiver: int = 10,
    vort_clip_quantiles: tuple[float, float] = (0.02, 0.98),
    mark_every_days: int = 7,
    mark_size: float = 45.0,
    markers: pd.DataFrame | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if not {"lat", "lon"}.issubset(df.columns):
        return

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    vort_col = pick_first_existing(df, ["vorticity"])
    wspd_col = pick_first_existing(df, ["wspd_mean", "wspd"])
    wdir_col = pick_first_existing(df, ["wdir_mean", "wdir"])

    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10.5, 7.5))
    ax = plt.axes(projection=proj)

    # Features
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)

    # Gridlines + labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.45)
    gl.top_labels = False
    gl.right_labels = False

    # Extent (auto with padding)
    mask = np.isfinite(lon) & np.isfinite(lat)
    if mask.any():
        lonm, latm = lon[mask], lat[mask]
        pad_lon = max(0.2, 0.05 * (np.nanmax(lonm) - np.nanmin(lonm) + 1e-12))
        pad_lat = max(0.2, 0.05 * (np.nanmax(latm) - np.nanmin(latm) + 1e-12))
        ax.set_extent(
            [
                np.nanmin(lonm) - pad_lon,
                np.nanmax(lonm) + pad_lon,
                np.nanmin(latm) - pad_lat,
                np.nanmax(latm) + pad_lat,
            ],
            crs=ccrs.PlateCarree(),
        )

    # Trajectory (colored by vorticity if available)
    if vort_col and df[vort_col].notna().any():
        vv = df[vort_col].to_numpy()
        vmin, vmax = np.nanquantile(vv, vort_clip_quantiles)
        sc = ax.scatter(
            lon,
            lat,
            c=vv,
            s=12,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(f"{vort_col} (1/s)")
    else:
        ax.plot(lon, lat, transform=ccrs.PlateCarree(), zorder=3)


    # Markers every N days (numbered)
    if "time_utc" in df.columns and mark_every_days and mark_every_days > 0:
        mk = (
            df[["time_utc", "lon", "lat"]]
            .dropna(subset=["time_utc", "lon", "lat"])
            .sort_values("time_utc")
            .set_index("time_utc")
            .resample(f"{int(mark_every_days)}D")
            .first()
            .dropna(subset=["lon", "lat"])
            .reset_index()
        )

    # Numbered markers (provided externally so they match TS)
    if markers is not None and not markers.empty:
        for r in markers.itertuples(index=False):
            ax.scatter(
                r.lon,
                r.lat,
                s=mark_size,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=1.2,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            ax.annotate(
                str(r.marker_id),
                xy=(r.lon, r.lat),
                xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=fontSize,
                weight="bold",
                ha="center",
                va="bottom",
                zorder=6,
            )


    # Start / End markers
    if "time_utc" in df.columns:
        df_sorted = df.dropna(subset=["lon", "lat", "time_utc"]).sort_values("time_utc")
        if not df_sorted.empty:
            start_row = df_sorted.iloc[0]
            end_row = df_sorted.iloc[-1]

            ax.scatter(
                start_row["lon"],
                start_row["lat"],
                marker="^",
                s=90,
                color="green",
                edgecolor="black",
                linewidth=0.8,
                transform=ccrs.PlateCarree(),
                zorder=6,
                label="Start",
            )
            ax.scatter(
                end_row["lon"],
                end_row["lat"],
                marker="X",
                s=90,
                color="red",
                edgecolor="black",
                linewidth=0.8,
                transform=ccrs.PlateCarree(),
                zorder=6,
                label="End",
            )

    # Wind arrows (decimated)
    if wspd_col and wdir_col:
        sub = df.iloc[:: max(1, decimate_quiver)].copy()
        sub = sub.dropna(subset=["lon", "lat", wspd_col, wdir_col])
        if len(sub) > 0:
            u, v = wind_dirspeed_to_uv(sub[wdir_col].to_numpy(), sub[wspd_col].to_numpy())
            ax.quiver(
                sub["lon"].to_numpy(),
                sub["lat"].to_numpy(),
                u,
                v,
                transform=ccrs.PlateCarree(),
                zorder=4,
                angles="xy",
                scale_units="xy",
                scale=25,
                width=0.0025,
                headwidth=3,
                headlength=4,
                headaxislength=3.5,
                alpha=0.8,
            )

    # Legend (only if something has labels)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper left", frameon=True, fontsize=fontSize)

    ax.set_title(f"{title_prefix}Trajectory")
    plt.tight_layout()
    fig.savefig(outdir / "map_trajectory_cartopy.png", dpi=200)
    plt.close(fig)



# ----------------------------
# Plot 2: wind rose (windrose package)
# ----------------------------
def plot_wind_rose_pkg(
    df: pd.DataFrame,
    outdir: Path,
    *,
    title_prefix: str = "",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    wspd_col = pick_first_existing(df, ["wspd_mean", "wspd"])
    wdir_col = pick_first_existing(df, ["wdir_mean", "wdir"])
    if not wspd_col or not wdir_col:
        return

    sub = df[[wspd_col, wdir_col]].dropna()
    if sub.empty:
        return

    from windrose import WindroseAxes

    fig = plt.figure(figsize=(7, 7))
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(
        sub[wdir_col].to_numpy(),
        sub[wspd_col].to_numpy(),
        normed=True,
        opening=0.9,
        edgecolor="white",
    )
    ax.set_title(f"{title_prefix}Wind rose ({wspd_col} vs {wdir_col})")
    ax.set_legend(title="m/s", loc="lower right", bbox_to_anchor=(1.15, 0.0))
    fig.savefig(outdir / "wind_rose.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Extra: quick gaps + availability report
# ----------------------------
def write_quick_report(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    t = df["time_utc"].sort_values()
    dt = t.diff().dt.total_seconds() / 3600.0
    gap_hours = dt[dt.notna() & (dt > 2)].to_numpy()

    cols = [
        "sst_c",
        "slp_mb",
        "salinity_psu",
        "rotation_index",
        "wspd",
        "wspd_mean",
        "wdir",
        "wdir_mean",
        "u",
        "v",
        "vorticity",
        "strain",
        "curvature_m1",
        "curvature_signed_m1",
        "okubo_weiss",
        "DO2_c",
        "oxy_comp_mgL_c",
        "bbp_470_m1",
        "bbp_532_m1",
    ]
    avail = {c: int(df[c].notna().sum()) for c in cols if c in df.columns}

    txt = []
    txt.append(f"rows: {len(df)}")
    if "platform_id" in df.columns and len(df) > 0:
        txt.append(f"platform_id: {df['platform_id'].iloc[0]}")
    txt.append(f"time span: {t.min()}  ->  {t.max()}")
    txt.append("")
    txt.append("availability (non-NaN counts):")
    for k, v in sorted(avail.items()):
        txt.append(f"  - {k}: {v}")
    txt.append("")
    txt.append("gaps > 2 hours (hours):")
    if gap_hours.size == 0:
        txt.append("  none")
    else:
        txt.append(f"  count: {gap_hours.size}")
        txt.append(f"  max:   {np.nanmax(gap_hours):.2f}")
        txt.append(f"  p95:   {np.nanpercentile(gap_hours, 95):.2f}")

    (outdir / "report.txt").write_text("\n".join(txt), encoding="utf-8")


# ----------------------------
# CLI
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Plot diagnostic figures from MASTER per-platform table.")
    ap.add_argument("--input", required=True, help="Path to master_<pid>.csv|parquet")
    ap.add_argument("--outdir", required=True, help="Output directory for figures")
    ap.add_argument("--decimate-quiver", type=int, default=10, help="Keep 1 every N points for wind arrows")
    ap.add_argument("--no-zero-sst", action="store_true", help="Treat SST==0 as NaN (fill-value behavior)")
    ap.add_argument("--sst-fill", default="-5.0", help="Comma-separated SST fill values to mask (e.g. -5, -999)")
    ap.add_argument("--smooth-all", default="", help="Time-based rolling mean window applied to ALL time series (e.g. '6H', '30min', '1D'). Empty disables.")
    ap.add_argument("--rot-smooth", default="", help="Time-based rolling mean window applied only to rotation_index (overrides --smooth-all for rotation). Empty disables.")
    ap.add_argument("--mark-every-days", type=int, default=7)
    ap.add_argument("--with-hourly-anomaly", action="store_true", help="Add hour-of-day anomaly on a right Y axis (y - mean(y|hour)).")
    ap.add_argument("--anom-min-samples-per-hour", type=int, default=3, help="Minimum samples per hour-of-day to compute the hour mean (otherwise anomaly is NaN for that hour).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    df = read_master(args.input)
    smooth_all = args.smooth_all.strip() or None
    rot_smooth = args.rot_smooth.strip() or None

    fill_vals = tuple(float(x.strip()) for x in str(args.sst_fill).split(",") if x.strip())
    df = apply_basic_masks(
        df,
        drop_sst_values=fill_vals,
        treat_zero_sst_as_nan=args.no_zero_sst,
    )

    title_prefix = ""
    if "platform_id" in df.columns and len(df) > 0:
        title_prefix = f"{df['platform_id'].iloc[0]} — "

    write_quick_report(df, outdir)

    markers = compute_time_markers(df, every_days=args.mark_every_days)
    anomaly_opts = AnomalyOpts(
        enabled=bool(args.with_hourly_anomaly),
        min_samples_per_hour=int(args.anom_min_samples_per_hour),
    )

    # Time series: single plots
    plot_time_series_singles(df, outdir, title_prefix=title_prefix, smooth_all=smooth_all, rot_smooth=rot_smooth, markers=markers, anomaly=anomaly_opts)

    # Cartopy map
    try:
        plot_trajectory_cartopy(df, outdir, title_prefix=title_prefix, decimate_quiver=args.decimate_quiver, markers=markers)
    except ModuleNotFoundError as e:
        print("Cartopy not installed. Install extras: pip install -e '.[viz]'")
        print(f"Reason: {e}")

    # Wind rose
    try:
        plot_wind_rose_pkg(df, outdir, title_prefix=title_prefix)
    except ModuleNotFoundError as e:
        print("windrose not installed. Install extras: pip install -e '.[viz]'")
        print(f"Reason: {e}")

    print(f"Wrote plots to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
Example:

python src/bgcd/plot_master.py `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065470010.csv" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots/300534065470010" `
  --no-zero-sst `
  --decimate-quiver 7 `
  --rot-smooth 2H `
  --smooth-all 6H
  --mark-every-days 7

python src/bgcd/plot_master.py `
   --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065378180.csv" `
   --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots/300534065378180" `
   --no-zero-sst `
   --decimate-quiver 7 `
   --smooth-all 12h `
   --rot-smooth 24h `
   --mark-every-days 7 `
   --with-hourly-anomaly `
   --anom-min-samples-per-hour 3
python src/bgcd/plot_master.py `
   --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065470010.csv" `
   --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots/300534065470010" `
   --no-zero-sst `
   --decimate-quiver 7 `
   --smooth-all 12h `
   --rot-smooth 24h `
   --mark-every-days 7 `
   --with-hourly-anomaly `
   --anom-min-samples-per-hour 3
python src/bgcd/plot_master.py `
   --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065379230.csv" `
   --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots/300534065379230" `
   --no-zero-sst `
   --decimate-quiver 7 `
   --smooth-all 12h `
   --rot-smooth 24h `
   --mark-every-days 7 `
   --with-hourly-anomaly `
   --anom-min-samples-per-hour 3

   


# nessuno smoothing
python src/bgcd/plot_master.py --input "...csv" --outdir "..."

# smoothing 6H su tutte le TS
python src/bgcd/plot_master.py --input "...csv" --outdir "..." --smooth-all 6H

# smoothing 6H su tutto, ma rotation più smooth (24H)
python src/bgcd/plot_master.py --input "...csv" --outdir "..." --smooth-all 6H --rot-smooth 24H

# smoothing 30 minuti (se hai dati molto fitti)
python src/bgcd/plot_master.py --input "...csv" --outdir "..." --smooth-all 30min
"""