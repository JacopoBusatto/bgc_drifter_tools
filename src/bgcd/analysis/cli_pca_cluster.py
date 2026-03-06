# ================================
# File: src/bgcd/analysis/cli_pca_cluster.py
# ================================
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.cluster import KMeans

from bgcd.analysis.cli_window import _load_yaml, _platform_id_from_df_or_filename


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_use_pcs(vals: List[int]) -> List[str]:
    pcs = []
    for v in vals:
        if v <= 0:
            raise ValueError("--use-pcs must be positive integers (1-based)")
        pcs.append(f"PC{int(v)}")
    return pcs


def _plot_pc_scatter(scores: pd.DataFrame, out_png: Path, title: str) -> None:
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111)

    sc = ax.scatter(scores["PC1"], scores["PC2"], c=scores["cluster"], s=26, alpha=0.9, edgecolor="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("cluster")

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_cluster_vs_time(scores: pd.DataFrame, out_png: Path, title: str) -> None:
    fig = plt.figure(figsize=(12.5, 3.8))
    ax = fig.add_subplot(111)

    t = pd.to_datetime(scores["time_utc"], utc=True, errors="coerce")
    ax.scatter(t, scores["cluster"], s=20, alpha=0.85, edgecolor="none")

    ax.set_yticks(sorted(scores["cluster"].dropna().unique()))
    ax.set_ylabel("cluster")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_trajectory_clusters(scores: pd.DataFrame, out_png: Path, title: str) -> None:
    if "lat" not in scores.columns or "lon" not in scores.columns:
        return

    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111)

    sc = ax.scatter(scores["lon"], scores["lat"], c=scores["cluster"], s=28, alpha=0.9, edgecolor="none")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(title)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("cluster")

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_ts_by_time_cluster(
    scores: pd.DataFrame,
    df_raw: pd.DataFrame,
    out_png: Path,
    title: str,
    smooth_window: str = "36h",
) -> None:
    """
    T-S diagram:
      - x = salinity
      - y = temperature
      - color = time
      - marker = PCA cluster
      - thin smoothed trajectory in T-S space
    """
    d = df_raw.copy()
    d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True, errors="coerce")

    s = scores[["time_utc", "cluster"]].copy()
    s["time_utc"] = pd.to_datetime(s["time_utc"], utc=True, errors="coerce")

    m = pd.merge(d, s, on="time_utc", how="inner")

    temp_col = None
    for c in ["temp_ctd_c", "sst_c"]:
        if c in m.columns:
            temp_col = c
            break

    sal_col = "salinity_psu" if "salinity_psu" in m.columns else None

    if temp_col is None or sal_col is None:
        return

    m[temp_col] = pd.to_numeric(m[temp_col], errors="coerce")
    m[sal_col] = pd.to_numeric(m[sal_col], errors="coerce")
    m = m.dropna(subset=["time_utc", temp_col, sal_col, "cluster"]).copy()

    if m.empty:
        return

    m = m.sort_values("time_utc").copy()
    m["time_num"] = mdates.date2num(m["time_utc"].dt.to_pydatetime())

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    clusters = sorted(m["cluster"].dropna().unique())

    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111)

    sc_for_cb = None

    for i, cl in enumerate(clusters):
        g = m[m["cluster"] == cl]
        mk = markers[i % len(markers)]

        sc = ax.scatter(
            g[sal_col],
            g[temp_col],
            c=g["time_num"],
            cmap="viridis",
            s=34,
            alpha=0.9,
            marker=mk,
            edgecolor="black",
            linewidth=0.25,
            label=f"cluster {int(cl)}",
        )
        sc_for_cb = sc

    # smoothed T-S trajectory using time-based rolling
    ts_src = (
        m[["time_utc", sal_col, temp_col]]
        .dropna()
        .sort_values("time_utc")
        .copy()
    )

    ts_line = (
        ts_src
        .rolling(smooth_window, on="time_utc", min_periods=3)[[sal_col, temp_col]]
        .mean()
    )
    ts_line["time_utc"] = ts_src["time_utc"].values
    ts_line = ts_line.dropna(subset=[sal_col, temp_col])

    if not ts_line.empty:
        ax.plot(
            ts_line[sal_col],
            ts_line[temp_col],
            linewidth=1.1,
            color="black",
            alpha=0.75,
            zorder=2,
            label=f"T-S smoothed path ({smooth_window})",
        )

    ax.set_xlabel("Salinity [PSU]")
    ax.set_ylabel(f"{temp_col} [°C]")
    ax.set_title(title)
    ax.grid(True, alpha=0.5)

    if sc_for_cb is not None:
        cb = fig.colorbar(sc_for_cb, ax=ax)
        cb.set_label("time")
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter("%Y-%m-%d")
        cb.ax.yaxis.set_major_locator(locator)
        cb.ax.yaxis.set_major_formatter(formatter)

    ax.legend(title="PCA cluster", loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _cluster_summary(df_raw: pd.DataFrame, scores: pd.DataFrame, vars_for_summary: List[str]) -> pd.DataFrame:
    """
    Join cluster labels back to the original subset (on time_utc) and compute summary stats per cluster.
    No interpolation: join is exact on time_utc.
    """
    d = df_raw.copy()
    d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True, errors="coerce")
    s = scores[["time_utc", "cluster"]].copy()
    s["time_utc"] = pd.to_datetime(s["time_utc"], utc=True, errors="coerce")

    m = pd.merge(d, s, on="time_utc", how="inner")

    cols = [c for c in vars_for_summary if c in m.columns]
    rows = []

    for cl in sorted(m["cluster"].dropna().unique()):
        g = m[m["cluster"] == cl]
        row: Dict[str, Any] = {"cluster": int(cl), "n": int(len(g))}
        for c in cols:
            v = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}__median"] = float(np.nanmedian(v.to_numpy())) if v.notna().any() else np.nan
            row[f"{c}__mean"] = float(np.nanmean(v.to_numpy())) if v.notna().any() else np.nan
            row[f"{c}__std"] = float(np.nanstd(v.to_numpy())) if v.notna().any() else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bgcd.analysis.cli_pca_cluster",
        description="KMeans clustering on PCA scores and plots trajectory colored by cluster. Text outputs -> OUT/<pid>/G_pca_cluster, figs -> plots/<pid>/analysis/pca_cluster.",
    )

    # Inputs
    ap.add_argument("--input", required=True, help="Same subset CSV used for PCA (best-window).")
    ap.add_argument("--config", required=True, help="analysis_config_min.yml")

    # Roots (match your structure)
    ap.add_argument("--out-root", required=True, help="Root OUT directory (e.g. .../OUT)")
    ap.add_argument("--plots-root", required=True, help="Root plots directory (e.g. .../plots)")

    # PCA scores location
    ap.add_argument("--pca-scores", default=None, help="Optional explicit path to pca_scores.csv. If omitted, uses plots/<pid>/analysis/pca/pca_scores.csv")

    # Clustering params
    ap.add_argument("--k", type=int, default=3, help="Number of clusters for KMeans.")
    ap.add_argument("--use-pcs", type=int, nargs="+", default=[1, 2], help="PC indices to use (e.g. --use-pcs 1 2 3).")
    ap.add_argument("--random-state", type=int, default=0)

    args = ap.parse_args()

    in_path = Path(args.input)
    cfg_path = Path(args.config)
    out_root = Path(args.out_root)
    plots_root = Path(args.plots_root)

    df_raw = pd.read_csv(in_path)
    pid = _platform_id_from_df_or_filename(df_raw, in_path)

    cfg = _load_yaml(cfg_path)
    analysis_cfg = (cfg or {}).get("analysis", {}) or {}
    phys_vars = analysis_cfg.get("phys_vars", []) or []
    bio_vars = analysis_cfg.get("bio_vars", []) or []
    summary_vars = list(dict.fromkeys(list(phys_vars) + list(bio_vars)))

    # locate PCA scores
    if args.pca_scores:
        scores_path = Path(args.pca_scores)
    else:
        scores_path = out_root / pid / "G_pca" / "pca_scores.csv"
    if not scores_path.exists():
        raise SystemExit(f"[ERR] pca_scores.csv not found: {scores_path}")

    scores = pd.read_csv(scores_path)
    scores["time_utc"] = pd.to_datetime(scores["time_utc"], utc=True, errors="coerce")
    scores = scores.dropna(subset=["time_utc"]).copy()

    pcs = _parse_use_pcs(args.use_pcs)
    for pc in pcs:
        if pc not in scores.columns:
            raise SystemExit(f"[ERR] {pc} not found in PCA scores columns.")

    # clustering matrix
    X = scores[pcs].to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1)
    scores = scores.loc[ok].copy()
    X = X[ok]

    km = KMeans(n_clusters=int(args.k), random_state=int(args.random_state), n_init="auto")
    scores["cluster"] = km.fit_predict(X)

    # outputs (match your structure)
    out_text_dir = out_root / pid / "H_pca_cluster"
    out_fig_dir = plots_root / pid / "analysis" / "pca_cluster"
    _ensure_dir(out_text_dir)
    _ensure_dir(out_fig_dir)

    # save cluster table (text)
    out_clusters_csv = out_text_dir / "pca_clusters.csv"
    scores.to_csv(out_clusters_csv, index=False)

    # cluster summary on original vars (text)
    summary = _cluster_summary(df_raw, scores, summary_vars)
    out_summary_csv = out_text_dir / "cluster_summary.csv"
    summary.to_csv(out_summary_csv, index=False)

    # plots (fig)
    if "PC1" in scores.columns and "PC2" in scores.columns:
        _plot_pc_scatter(scores, out_fig_dir / "pc_scatter_clusters.png", title=f"{pid} | PCA scores colored by cluster (k={args.k})")
    _plot_cluster_vs_time(scores, out_fig_dir / "cluster_vs_time.png", title=f"{pid} | cluster vs time (k={args.k})")
    _plot_trajectory_clusters(scores, out_fig_dir / "trajectory_clusters.png", title=f"{pid} | trajectory colored by cluster (k={args.k})")
    _plot_ts_by_time_cluster(scores, df_raw, out_fig_dir / "ts_diagram_time_cluster.png", title=f"{pid} | T-S diagram colored by time, marker by cluster (k={args.k})")
    # log (text)
    lines = [
        f"INPUT: {in_path}",
        f"CONFIG: {cfg_path}",
        f"PLATFORM_ID: {pid}",
        f"PCA_SCORES: {scores_path}",
        "",
        f"k: {args.k}",
        f"use_pcs: {pcs}",
        f"random_state: {args.random_state}",
        f"n_points_used: {len(scores)}",
        "",
        "text outputs:",
        f"  - {out_clusters_csv}",
        f"  - {out_summary_csv}",
        "",
        "figure outputs:",
        f"  - {out_fig_dir / 'pc_scatter_clusters.png'}",
        f"  - {out_fig_dir / 'cluster_vs_time.png'}",
        f"  - {out_fig_dir / 'trajectory_clusters.png'}",
    ]
    _write_text(out_text_dir / "run_log.txt", lines)

    print(f"[OK] platform_id={pid}")
    print(f"[OK] text outputs: {out_text_dir}")
    print(f"[OK] figures:      {out_fig_dir}")


if __name__ == "__main__":
    main()

"""
python -m bgcd.analysis.cli_pca_cluster `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065378180/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min_no_sal.yml" `
  --out-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --k 3 `
  --use-pcs 1 2 3

python -m bgcd.analysis.cli_pca_cluster `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065379230/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --out-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --k 3 `
  --use-pcs 1 2 3

python -m bgcd.analysis.cli_pca_cluster `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065470010/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --out-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --k 3 `
  --use-pcs 1 2 3
"""