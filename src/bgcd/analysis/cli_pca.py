# ================================
# File: src/bgcd/analysis/cli_pca.py
# ================================
from __future__ import annotations


import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from bgcd.analysis.pca import run_pca
from bgcd.analysis.cli_window import _load_yaml, _dedup_keep_order, _platform_id_from_df_or_filename


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _plot_variance(explained: pd.DataFrame, out_png: Path, title: str, max_pcs: int = 10) -> None:
    d = explained.copy()
    if len(d) > max_pcs:
        d = d.iloc[:max_pcs].copy()

    fig = plt.figure(figsize=(10.2, 4.2))
    ax = fig.add_subplot(111)

    ax.bar(d["PC"], d["explained_var_ratio"])
    ax.plot(d["PC"], d["explained_var_ratio_cum"], marker="o")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Explained variance ratio (bar) / cumulative (line)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_scores_timeseries(scores: pd.DataFrame, out_png: Path, title: str, pcs: List[str]) -> None:
    fig = plt.figure(figsize=(12.5, 6.0))
    ax = fig.add_subplot(111)

    t = pd.to_datetime(scores["time_utc"], utc=True, errors="coerce")
    for pc in pcs:
        if pc in scores.columns:
            ax.plot(t, scores[pc], label=pc)

    ax.axhline(0.0, linewidth=0.9, alpha=0.7)
    ax.set_ylabel("PC score (z-space)")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)




def _plot_scatter_time_colored(scores, out_png, title, pcx="PC1", pcy="PC2"):

    t = pd.to_datetime(scores["time_utc"], utc=True, errors="coerce")

    # datetime → matplotlib date numbers
    tn = mdates.date2num(t)

    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111)

    sc = ax.scatter(
        scores[pcx],
        scores[pcy],
        c=tn,
        s=24,
        alpha=0.9,
        edgecolor="none",
    )

    ax.set_xlabel(pcx)
    ax.set_ylabel(pcy)
    ax.set_title(title)

    cb = fig.colorbar(sc, ax=ax)
    cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    cb.set_label("time (UTC)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def _plot_trajectory_colored_by_pc(scores: pd.DataFrame, out_png: Path, title: str, pc: str = "PC1") -> None:
    if "lat" not in scores.columns or "lon" not in scores.columns:
        return
    if pc not in scores.columns:
        return

    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111)

    # Use robust scaling for color (avoid extreme outliers dominating)
    v = scores[pc].to_numpy(dtype=float)
    vmin = np.nanquantile(v, 0.02)
    vmax = np.nanquantile(v, 0.98)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        scores["lon"],
        scores["lat"],
        c=v,
        norm=norm,
        s=22,
        alpha=0.9,
        edgecolor="none",
        cmap="coolwarm",
    )

    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(title)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(pc)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def _plot_biplot(loadings: pd.DataFrame, scores: pd.DataFrame, out_png: Path, title: str, pcx: str = "PC1", pcy: str = "PC2") -> None:
    # Biplot: scores cloud + loading vectors
    fig = plt.figure(figsize=(6., 5.))
    ax = fig.add_subplot(111)

    # scores cloud (grey)
    ax.scatter(scores[pcx], scores[pcy], s=20, alpha=0.5)

    # scale arrows to look nice
    # take a typical score radius
    sx = np.nanstd(scores[pcx])
    sy = np.nanstd(scores[pcy])
    scale = 0.8 * min(sx, sy) if (sx > 0 and sy > 0) else 1.0

    # choose top variables by loading magnitude on PC1-PC2
    d = loadings[[pcx, pcy]].copy()
    d["mag"] = np.sqrt(d[pcx] ** 2 + d[pcy] ** 2)
    d = d.sort_values("mag", ascending=False)

    topk = 6  # cambia tu: 6-10 di solito ok
    for var in d.index[:topk]:
        vx = float(loadings.loc[var, pcx]) * scale
        vy = float(loadings.loc[var, pcy]) * scale
        ax.arrow(0, 0, vx, vy, head_width=0.04 * scale, length_includes_head=True, alpha=0.9)
        ax.text(vx * 1.08, vy * 1.08, var, fontsize=9)

    ax.axhline(0.0, linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, linewidth=0.8, alpha=0.6)
    ax.set_xlabel(f"{pcx} (scores)")
    ax.set_ylabel(f"{pcy} (scores)")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bgcd.analysis.cli_pca",
        description="PCA on best-window subset (no interpolation). Outputs: variance, loadings, scores, biplot, time-colored scatter, PC time series.",
    )
    ap.add_argument("--input", required=True, help="CSV to analyze (recommended: best-window subset).")
    ap.add_argument("--config", required=True, help="analysis_config_min.yml")
    ap.add_argument("--outdir", default="OUT", help="Base output directory for non-plot outputs (like CSVs). Will write: <outdir>/<pid>/analysis/pca/ ...")
    ap.add_argument("--plots-root", default="plots", help="Root folder: <root>/<pid>/analysis/pca/ ...")
    ap.add_argument("--n-components", type=int, default=None, help="Optional number of PCs to keep (default: all).")
    ap.add_argument("--max-pcs-variance-plot", type=int, default=10, help="Max PCs shown in variance plot.")
    args = ap.parse_args()

    in_path = Path(args.input)
    cfg_path = Path(args.config)
    plots_root = Path(args.plots_root)
    out_base = Path(args.outdir)

    cfg = _load_yaml(cfg_path)
    analysis_cfg: Dict[str, Any] = (cfg or {}).get("analysis", {}) or {}
    phys_vars = _dedup_keep_order(analysis_cfg.get("phys_vars", []) or [])
    bio_vars = _dedup_keep_order(analysis_cfg.get("bio_vars", []) or [])
    variables = _dedup_keep_order(phys_vars + bio_vars)

    df = pd.read_csv(in_path)
    pid = _platform_id_from_df_or_filename(df, in_path)

    out_csv_dir = out_base / pid / "G_pca"
    out_fig_dir = plots_root / pid / "analysis" / "pca"
    _ensure_dir(out_csv_dir)
    _ensure_dir(out_fig_dir)

    # run PCA
    res = run_pca(
        df,
        variables=variables,
        time_col="time_utc",
        keep_cols=[c for c in ["lat", "lon"] if c in df.columns],
        n_components=args.n_components,
    )

    # write tables
    res.loadings.to_csv(out_csv_dir / "pca_loadings.csv")
    res.scores.to_csv(out_csv_dir / "pca_scores.csv", index=False)
    res.explained.to_csv(out_csv_dir / "pca_explained_variance.csv", index=False)

    # plots
    _plot_variance(
        res.explained,
        out_fig_dir / "pca_variance.png",
        title=f"{pid} | PCA explained variance (n_used={res.n_rows_used}/{res.n_rows_in})",
        max_pcs=int(args.max_pcs_variance_plot),
    )

    if "PC1" in res.scores.columns and "PC2" in res.scores.columns:
        _plot_scatter_time_colored(
            res.scores,
            out_fig_dir / "pca_scatter_PC1_PC2_time.png",
            title=f"{pid} | PCA scores PC1 vs PC2 (colored by time)",
            pcx="PC1",
            pcy="PC2",
        )
        _plot_biplot(
            res.loadings,
            res.scores,
            out_fig_dir / "pca_biplot_PC1_PC2.png",
            title=f"{pid} | PCA biplot PC1–PC2 (loadings vectors)",
            pcx="PC1",
            pcy="PC2",
        )

    # Trajectory colored by PC1 and PC2 (if lat/lon available)
    if "lat" in res.scores.columns and "lon" in res.scores.columns:
        if "PC1" in res.scores.columns:
            _plot_trajectory_colored_by_pc(
                res.scores,
                out_fig_dir / "pca_trajectory_colored_PC1.png",
                title=f"{pid} | trajectory colored by PC1",
                pc="PC1",
            )
        if "PC2" in res.scores.columns:
            _plot_trajectory_colored_by_pc(
                res.scores,
                out_fig_dir / "pca_trajectory_colored_PC2.png",
                title=f"{pid} | trajectory colored by PC2",
                pc="PC2",
            )

    pcs_ts = [pc for pc in ["PC1", "PC2", "PC3"] if pc in res.scores.columns]
    if pcs_ts:
        _plot_scores_timeseries(
            res.scores,
            out_fig_dir / "pca_scores_timeseries.png",
            title=f"{pid} | PCA scores vs time",
            pcs=pcs_ts,
        )

    # run log (super utile per debug/repro)
    lines = [
        f"INPUT: {in_path}",
        f"CONFIG: {cfg_path}",
        f"PLATFORM_ID: {pid}",
        "",
        f"n_rows_in: {res.n_rows_in}",
        f"n_rows_used (dropna on PCA vars): {res.n_rows_used}",
        "",
        f"variables ({len(res.variables)}): {res.variables}",
        "",
        "outputs:",
        f"  - {out_csv_dir / 'pca_loadings.csv'}",
        f"  - {out_csv_dir / 'pca_scores.csv'}",
        f"  - {out_csv_dir / 'pca_explained_variance.csv'}",
        f"  - {out_fig_dir / 'pca_variance.png'}",
        f"  - {out_fig_dir / 'pca_scatter_PC1_PC2_time.png'}",
        f"  - {out_fig_dir / 'pca_biplot_PC1_PC2.png'}",
        f"  - {out_fig_dir / 'pca_scores_timeseries.png'}",
    ]
    _write_text(out_csv_dir / "run_log.txt", lines)

    print(f"[OK] platform_id={pid}")
    print(f"[OK] PCA outputs in: {out_csv_dir}")


if __name__ == "__main__":
    main()
"""
python -m bgcd.analysis.cli_pca `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/OUT/300534065379230/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots"
  
"""