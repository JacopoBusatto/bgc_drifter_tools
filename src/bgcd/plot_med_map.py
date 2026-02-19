#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        raise ValueError(f"{path.name}: expected column 'time_utc'")
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")

    # platform_id
    if "platform_id" in df.columns:
        df["platform_id"] = df["platform_id"].astype(str).str.strip()
    else:
        # fallback: use filename stem
        df["platform_id"] = path.stem.replace("master_", "")

    # numeric conversion (keep platform_id + time_utc)
    for c in df.columns:
        if c in ("platform_id", "time_utc"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    return df


def normalize_lon180(lon: np.ndarray) -> np.ndarray:
    """Convert longitudes to [-180, 180] for consistent Mediterranean plotting."""
    lon = np.asarray(lon, dtype=float)
    out = lon.copy()
    out[out > 180] -= 360
    return out


def plot_med_map(
    dfs: list[pd.DataFrame],
    outpath: Path,
    *,
    title: str = "Mediterranean drifter trajectories",
    extent: tuple[float, float, float, float] = (-7.0, 37.0, 30.0, 46.5),  # lon_min, lon_max, lat_min, lat_max
    linewidth: float = 2.0,
    alpha: float = 0.95,
    show_points: bool = False,
    point_size: float = 6.0,
) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(12.5, 8.0))
    ax = plt.axes(projection=proj)

    # base map
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.9, zorder=1)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.6, zorder=1)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.45)
    gl.top_labels = False
    gl.right_labels = False

    # plot each drifter
    for df in dfs:
        if not {"lon", "lat"}.issubset(df.columns):
            continue

        dfp = (
            df[["platform_id", "time_utc", "lon", "lat"]]
            .dropna(subset=["time_utc", "lon", "lat"])
            .sort_values("time_utc")
        )
        if dfp.empty:
            continue

        pid = str(dfp["platform_id"].iloc[0])

        lon = normalize_lon180(dfp["lon"].to_numpy())
        lat = dfp["lat"].to_numpy()

        # main line
        (line,) = ax.plot(
            lon,
            lat,
            transform=proj,
            linewidth=linewidth,
            alpha=alpha,
            label=pid,
            zorder=3,
        )

        # optional points
        if show_points:
            ax.scatter(
                lon,
                lat,
                transform=proj,
                s=point_size,
                alpha=0.35,
                zorder=3,
            )

        # start / end markers (same color as line)
        c = line.get_color()
        ax.scatter(
            lon[0],
            lat[0],
            transform=proj,
            marker="^",
            s=110,
            color=c,
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.scatter(
            lon[-1],
            lat[-1],
            transform=proj,
            marker="X",
            s=110,
            color=c,
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True, fontsize=10, framealpha=0.9)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot a full Mediterranean map with trajectories of 3 drifters (or more)."
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to master_<pid>.csv or .parquet (provide 3 files, but can be more).",
    )
    ap.add_argument("--out", required=True, help="Output PNG path (e.g. plots/med_map_3drifters.png)")
    ap.add_argument("--title", default="Mediterranean drifter trajectories")
    ap.add_argument("--extent", default="-7,37,30,46.5", help="lonmin,lonmax,latmin,latmax")
    ap.add_argument("--linewidth", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--show-points", action="store_true", help="Also scatter the raw points.")
    ap.add_argument("--point-size", type=float, default=6.0)
    args = ap.parse_args()

    extent = tuple(float(x.strip()) for x in args.extent.split(","))
    if len(extent) != 4:
        raise ValueError("--extent must be lonmin,lonmax,latmin,latmax")

    dfs = [read_master(p) for p in args.inputs]
    outpath = Path(args.out)

    plot_med_map(
        dfs,
        outpath,
        title=args.title,
        extent=extent,  # full Mediterranean by default
        linewidth=args.linewidth,
        alpha=args.alpha,
        show_points=bool(args.show_points),
        point_size=args.point_size,
    )

    print(f"Wrote: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
