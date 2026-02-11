#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def _add_right_axis(ax, offset: float):
    """Create an additional y-axis on the right, offset outward by `offset`."""
    axr = ax.twinx()
    axr.spines["right"].set_position(("outward", offset))
    axr.spines["right"].set_visible(True)
    return axr


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
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if not {"lat", "lon"}.issubset(df.columns):
        return

    # imports are inside so script still runs without viz extras (it will just fail here if called)
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

    # Trajectory (colored)
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
                scale=25,          # aumenta → frecce più corte
                width=0.005,      # ↓ più piccolo = più fine
                headwidth=3,
                headlength=4,
                headaxislength=3.5,
                alpha=0.8,
            )

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


def plot_time_series_pair(
    df: pd.DataFrame,
    outpath: Path,
    *,
    title: str,
    y1_col: str,
    y1_label: str,
    y2_col: str | None = None,
    y2_label: str | None = None,
    time_col: str = "time_utc",
    smooth_points: int = 0,
) -> None:
    """
    Plot two variables on the same time axis, using a secondary y-axis if needed.
    Saves a single PNG to outpath.
    """
    dff = df.copy()

    cols_to_smooth = [y1_col] + ([y2_col] if y2_col else [])
    cols_to_smooth = [c for c in cols_to_smooth if c and c in dff.columns]

    if smooth_points and smooth_points > 1:
        for c in cols_to_smooth:
            dff[c] = dff[c].rolling(smooth_points, min_periods=1).mean()

    t = dff[time_col]

    fig = plt.figure(figsize=(12.5, 5.2))
    ax1 = plt.gca()
    ax1.grid(True, which="both", alpha=0.35)

    # y1
    if y1_col not in dff.columns:
        plt.close(fig)
        return
    l1 = ax1.plot(t, dff[y1_col], linewidth=1.4, label=y1_label)[0]
    ax1.set_ylabel(y1_label)

    lines = [l1]
    labels = [y1_label]

    # y2
    if y2_col and (y2_col in dff.columns):
        ax2 = ax1.twinx()
        l2 = ax2.plot(t, dff[y2_col], linewidth=1.2, linestyle="--", label=y2_label or y2_col)[0]
        ax2.set_ylabel(y2_label or y2_col)
        lines.append(l2)
        labels.append(y2_label or y2_col)
    else:
        ax2 = None

    ax1.set_xlabel("Time (UTC)")
    ax1.set_title(title)

    # compact legend
    ax1.legend(lines, labels, loc="upper left", frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_time_series_pairs(
    df: pd.DataFrame,
    outdir: Path,
    *,
    title_prefix: str = "",
    smooth_points: int = 0,
) -> None:
    """
    Produce a small set of readable paired time series plots.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # choose available columns
    sst = "sst_c" if "sst_c" in df.columns else None
    sal = "salinity_psu" if "salinity_psu" in df.columns else None
    vort = "vorticity" if "vorticity" in df.columns else None
    rot = "rotation_index" if "rotation_index" in df.columns else None
    wspd = pick_first_existing(df, ["wspd_mean", "wspd"])

    # 1) SST + Salinity
    if sst and sal:
        plot_time_series_pair(
            df,
            outdir / "ts_sst_salinity.png",
            title=f"{title_prefix}SST & Salinity",
            y1_col=sst,
            y1_label="SST (°C)",
            y2_col=sal,
            y2_label="Salinity (PSU)",
            smooth_points=smooth_points,
        )
    elif sst:
        plot_time_series_pair(
            df,
            outdir / "ts_sst.png",
            title=f"{title_prefix}SST",
            y1_col=sst,
            y1_label="SST (°C)",
            smooth_points=smooth_points,
        )
    elif sal:
        plot_time_series_pair(
            df,
            outdir / "ts_salinity.png",
            title=f"{title_prefix}Salinity",
            y1_col=sal,
            y1_label="Salinity (PSU)",
            smooth_points=smooth_points,
        )

    # 2) Wind speed + Rotation index
    if wspd and rot:
        plot_time_series_pair(
            df,
            outdir / "ts_wind_rotation.png",
            title=f"{title_prefix}Wind speed & Rotation index",
            y1_col=wspd,
            y1_label="Wind speed (m/s)",
            y2_col=rot,
            y2_label="Rotation index (-)",
            smooth_points=smooth_points,
        )
    elif wspd:
        plot_time_series_pair(
            df,
            outdir / "ts_wind.png",
            title=f"{title_prefix}Wind speed",
            y1_col=wspd,
            y1_label="Wind speed (m/s)",
            smooth_points=smooth_points,
        )
    elif rot:
        plot_time_series_pair(
            df,
            outdir / "ts_rotation.png",
            title=f"{title_prefix}Rotation index",
            y1_col=rot,
            y1_label="Rotation index (-)",
            smooth_points=smooth_points,
        )

    # 3) Vorticity + Rotation index (optional but useful)
    if vort and rot:
        plot_time_series_pair(
            df,
            outdir / "ts_vorticity_rotation.png",
            title=f"{title_prefix}Vorticity & Rotation index",
            y1_col=vort,
            y1_label="Vorticity (1/s)",
            y2_col=rot,
            y2_label="Rotation index (-)",
            smooth_points=smooth_points,
        )
    elif vort:
        plot_time_series_pair(
            df,
            outdir / "ts_vorticity.png",
            title=f"{title_prefix}Vorticity",
            y1_col=vort,
            y1_label="Vorticity (1/s)",
            smooth_points=smooth_points,
        )





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
    ap.add_argument("--smooth", type=int, default=0, help="Rolling mean window (points) for time series (0 disables)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    df = read_master(args.input)

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

    # 1) Single time-series figure
    plot_time_series_pairs(df, outdir, title_prefix=title_prefix, smooth_points=args.smooth)

    # 2) Cartopy map
    try:
        plot_trajectory_cartopy(df, outdir, title_prefix=title_prefix, decimate_quiver=args.decimate_quiver)
    except ModuleNotFoundError as e:
        print("Cartopy not installed. Install extras: pip install -e '.[viz]'")
        print(f"Reason: {e}")

    # 3) Wind rose (windrose package)
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
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065378180.csv" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots/300534065378180" `
  --no-zero-sst `
  --decimate-quiver 10 `
  --smooth 0
"""