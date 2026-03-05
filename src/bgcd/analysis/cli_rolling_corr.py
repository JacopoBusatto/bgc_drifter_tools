# ================================
# File: src/bgcd/analysis/cli_rolling_corr.py
# ================================
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bgcd.analysis.rolling_corr import rolling_corr_many
from bgcd.analysis.cli_window import _load_yaml, _dedup_keep_order, _platform_id_from_df_or_filename


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _pick_default_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Default set (minimal but scientific) of rolling pairs.
    We keep it small to avoid noise; you can override via --pairs.
    """
    candidates = [
        ("salinity_psu", "DO2_c"),
        ("salinity_psu", "oxy_comp_mgL_c"),  # if DO2_c missing but mg/L exists
        ("wspd", "DO2_c"),
        ("wspd_mean", "DO2_c"),
        ("sst_c", "bbp_532_m1"),
        ("sst_c", "chl"),
        ("okubo_weiss", "chl"),
        ("okubo_weiss", "bbp_532_m1"),
    ]
    out = []
    for x, y in candidates:
        if x in df.columns and y in df.columns:
            out.append((x, y))
    # de-dup exact duplicates
    seen = set()
    out2 = []
    for p in out:
        if p not in seen:
            out2.append(p)
            seen.add(p)
    return out2


def _parse_pairs_arg(pairs_str: str) -> List[Tuple[str, str]]:
    """
    Format:
      "x1:y1,x2:y2, ..."
    """
    s = (pairs_str or "").strip()
    if not s:
        return []
    pairs = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad --pairs item '{item}'. Use 'x:y'.")
        x, y = item.split(":", 1)
        pairs.append((x.strip(), y.strip()))
    return pairs


def _plot_multiline_rolling(
    series_list,
    out_png: Path,
    *,
    title: str,
) -> None:
    fig = plt.figure(figsize=(12.5, 4.6))
    ax = fig.add_subplot(111)

    for s in series_list:
        d = s.df
        ax.plot(d["time_utc"], d["r"], label=s.pair_name)

    ax.axhline(0.0, linewidth=0.9)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("rolling Pearson r")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_rolling_with_npairs(
    series,
    out_png: Path,
    *,
    title: str,
) -> None:
    d = series.df

    fig = plt.figure(figsize=(12.5, 5.2))
    ax1 = fig.add_subplot(111)
    ax1.plot(d["time_utc"], d["r"])
    # mark significant points if p_value exists
    if "p_value" in d.columns:
        sig = d["p_value"].notna() & (d["p_value"] < 0.05) & d["r"].notna()
        if sig.any():
            ax1.scatter(d.loc[sig, "time_utc"], d.loc[sig, "r"], s=12, alpha=0.8, label="p<0.05")
            ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.axhline(0.0, linewidth=0.9)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_ylabel("rolling Pearson r")
    ax1.set_title(title)

    # n_pairs on right axis
    ax2 = ax1.twinx()
    ax2.plot(d["time_utc"], d["n_pairs"], alpha=0.4)
    ax2.set_ylabel("n_pairs in window")

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bgcd.analysis.cli_rolling_corr",
        description="Rolling correlation (time-based window, irregular sampling, no interpolation).",
    )
    ap.add_argument(
        "--input",
        required=True,
        help="CSV to analyze (recommended: OUT/<pid>/B_window/data/master_subset_best_window.csv).",
    )
    ap.add_argument("--config", required=True, help="analysis_config_min.yml")
    ap.add_argument(
        "--plots-root",
        default="plots",
        help="Root folder for plot outputs. Will write: <plots-root>/<pid>/analysis/rolling/ ...",
    )
    ap.add_argument(
        "--outdir",
        default="OUT",
        help="Base output directory for non-plot outputs (like CSVs). Will write: <outdir>/<pid>/analysis/rolling/ ...",
    )
    ap.add_argument(
        "--window-hours",
        type=int,          # <-- ERA float
        default=48,
        help="Rolling window in hours (INTEGER, time-based). Example: 48",
    )
    ap.add_argument(
        "--center",
        action="store_true",
        help="Use centered window [t-W/2, t+W/2]. Default is centered.",
    )
    ap.add_argument(
        "--trailing",
        action="store_true",
        help="Use trailing window [t-W, t]. If set, overrides --center.",
    )
    ap.add_argument(
        "--min-points",
        type=int,
        default=10,
        help="Minimum valid (x,y) pairs inside a rolling window to compute r.",
    )
    ap.add_argument(
        "--pairs",
        default="",
        help="Optional pairs to compute: 'x1:y1,x2:y2'. If empty, uses a small default set.",
    )
    ap.add_argument(
        "--with-trajectory-map",
        action="store_true",
        help="Also write map_trajectory_cartopy.png for THIS WINDOW using plot_master.plot_trajectory_cartopy().",
    )
    ap.add_argument(
        "--decimate-quiver",
        type=int,
        default=10,
        help="Decimation factor for wind arrows in the trajectory map.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    cfg_path = Path(args.config)
    plots_root = Path(args.plots_root)
    out_base = Path(args.outdir)

    cfg = _load_yaml(cfg_path)
    analysis_cfg: Dict[str, Any] = (cfg or {}).get("analysis", {}) or {}

    df = pd.read_csv(in_path)
    pid = _platform_id_from_df_or_filename(df, in_path)

    out_csv_dir = out_base / pid / "E_rolling"
    out_fig_dir = plots_root / pid / "analysis" / "rolling"

    _ensure_dir(out_csv_dir)
    _ensure_dir(out_fig_dir)

    # choose pairs
    pairs = _parse_pairs_arg(args.pairs)
    if not pairs:
        pairs = _pick_default_pairs(df)

    center = True
    if bool(args.trailing):
        center = False

    # e nella chiamata:
    series_list = rolling_corr_many(
        df,
        pairs=pairs,
        time_col="time_utc",
        window_hours=int(args.window_hours),
        min_points=int(args.min_points),
        center=center,
    )

    # compute
    series_list = rolling_corr_many(
        df,
        pairs=pairs,
        time_col="time_utc",
        window_hours=int(args.window_hours),
        min_points=int(args.min_points),
    )

    if not series_list:
        raise SystemExit("No rolling series computed (pairs missing or columns not present).")

    # write per-pair CSV + per-pair plot (with n_pairs)
    for s in series_list:
        out_csv = out_csv_dir / f"rolling_{s.pair_name}.csv"
        s.df.to_csv(out_csv, index=False)

        out_png = out_fig_dir / f"rolling_{s.pair_name}.png"
        _plot_rolling_with_npairs(
            s,
            out_png,
            title=f"{pid} | rolling corr {s.pair_name} | window={args.window_hours}h | min_points={args.min_points}",
        )

    # # write multiline summary plot
    # out_png_all = out_fig_dir / "rolling_all_pairs.png"
    # _plot_multiline_rolling(
    #     series_list,
    #     out_png_all,
    #     title=f"{pid} | rolling corr (all pairs) | window={args.window_hours}h | min_points={args.min_points}",
    # )

    # optional: trajectory map for THIS window
    if bool(args.with_trajectory_map):
        # reuse plot_master without modifying it
        # (map will reflect the filtered window, because df is already the window subset)
        try:
            from bgcd.plot_master import compute_time_markers, plot_trajectory_cartopy  # re-use existing code :contentReference[oaicite:1]{index=1}

            # ensure datetime for plot_master helpers (they rely on resample)
            df_map = df.copy()
            df_map["time_utc"] = pd.to_datetime(df_map["time_utc"], errors="coerce", utc=True)
            df_map = df_map.loc[df_map["time_utc"].notna()].copy()

            markers = compute_time_markers(df_map, every_days=7)
            plot_trajectory_cartopy(
                df_map,
                out_fig_dir,
                title_prefix=f"{pid} — window — ",
                decimate_quiver=int(args.decimate_quiver),
                markers=markers,
            )
        except ModuleNotFoundError as e:
            print("Cartopy not installed; skipping trajectory map.")
            print(f"Reason: {e}")

    # run log
    lines = [
        f"INPUT: {in_path}",
        f"CONFIG: {cfg_path}",
        f"PLATFORM_ID: {pid}",
        "",
        f"window_hours: {args.window_hours}",
        f"min_points: {args.min_points}",
        "",
        "pairs:",
        *[f"  - {x} : {y}" for x, y in pairs],
        "",
        f"wrote per-pair CSVs: {out_csv_dir}",
        f"wrote per-pair figs: {out_fig_dir}",
        # f"wrote summary fig: {out_png_all}",
        f"trajectory map: {'yes' if args.with_trajectory_map else 'no'}",
    ]
    _write_text(out_csv_dir / "run_log.txt", lines)

    print(f"[OK] platform_id={pid}")
    print(f"[OK] wrote rolling outputs in: {out_csv_dir} and {out_fig_dir}")


if __name__ == "__main__":
    main()

"""

python -m bgcd.analysis.cli_rolling_corr `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065378180/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --window-hours 48 `
  --min-points 10 `
  --with-trajectory-map `
  --decimate-quiver 7

python -m bgcd.analysis.cli_rolling_corr `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065379230/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --window-hours 48 `
  --min-points 10 `
  --with-trajectory-map `
  --decimate-quiver 7

python -m bgcd.analysis.cli_rolling_corr `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065470010/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --window-hours 48 `
  --min-points 10 `
  --with-trajectory-map `
  --decimate-quiver 7
"""