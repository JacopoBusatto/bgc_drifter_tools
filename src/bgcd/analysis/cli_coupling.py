# ================================
# File: src/bgcd/analysis/cli_coupling.py
# ================================
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bgcd.analysis.coupling import pairwise_pearson_phys_bio
from bgcd.analysis.cli_window import _load_yaml, _dedup_keep_order, _platform_id_from_df_or_filename  # reuse helpers


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _p_to_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _make_heatmap(table: pd.DataFrame, phys: List[str], bio: List[str], out_png: Path, title: str) -> None:
    # build matrix phys x bio
    M = np.full((len(phys), len(bio)), np.nan, dtype=float)
    P = np.full((len(phys), len(bio)), np.nan, dtype=float)

    for _, row in table.iterrows():
        if row.get("status") != "ok":
            continue

        x = row["x"]
        y = row["y"]

        if x in phys and y in bio:
            i = phys.index(x)
            j = bio.index(y)

            M[i, j] = float(row["pearson_r"])

            if "p_value" in row:
                P[i, j] = float(row["p_value"])

    fig = plt.figure(figsize=(max(6, 0.55 * len(bio)), max(4, 0.45 * len(phys))))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto", vmin=-1, vmax=1)

    # annotate cells with r and significance
    for i in range(len(phys)):
        for j in range(len(bio)):

            r = M[i, j]
            if np.isnan(r):
                continue

            p = P[i, j]
            stars = _p_to_stars(p)

            txt = f"{r:.2f}{stars}"

            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    ax.set_title(title)

    ax.set_yticks(np.arange(len(phys)))
    ax.set_yticklabels(phys)
    ax.set_xticks(np.arange(len(bio)))
    ax.set_xticklabels(bio, rotation=45, ha="right")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Pearson r")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bgcd.analysis.cli_coupling",
        description="Pairwise coupling (Pearson) phys×bio on a windowed MASTER subset (no interpolation).",
    )
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV to analyze (e.g., OUT/<pid>/B_window/data/master_subset_best_window.csv).",
    )
    ap.add_argument("--config", type=str, required=True, help="Path to analysis_config_min.yml.")
    ap.add_argument("--outdir", type=str, default="OUT", help="Base output directory. Outputs are isolated in OUT/<platform_id>/D_coupling/...")
    ap.add_argument("--plots-root", type=str, default="plots", help="Root folder for plots: <root>/<pid>/analysis/coupling/ ...")

    args = ap.parse_args()

    in_path = Path(args.input)
    cfg_path = Path(args.config)
    out_base = Path(args.outdir)

    cfg = _load_yaml(cfg_path)
    analysis_cfg = (cfg or {}).get("analysis", {}) or {}
    phys_vars = _dedup_keep_order(analysis_cfg.get("phys_vars", []) or [])
    bio_vars = _dedup_keep_order(analysis_cfg.get("bio_vars", []) or [])

    qc_cfg = (analysis_cfg.get("qc", {}) or {})
    po = (qc_cfg.get("pairwise_overlap", {}) or {})
    min_fraction = float(po.get("min_fraction", 0.7))
    min_points = int(po.get("min_points", 50))

    df = pd.read_csv(in_path)
    pid = _platform_id_from_df_or_filename(df, in_path)

    plots_root = Path(args.plots_root)

    out_platform = out_base / pid
    out_coup = out_platform / "D_coupling" / "pairwise"
    out_reports = out_coup / "reports"
    out_figs = plots_root / pid / "analysis" / "pairwise_coupling"

    _ensure_dir(out_reports)
    _ensure_dir(out_figs)

    res = pairwise_pearson_phys_bio(
        df,
        phys_vars=phys_vars,
        bio_vars=bio_vars,
        min_fraction=min_fraction,
        min_points=min_points,
    )

    out_csv = out_reports / "pairwise_corr.csv"
    res.table.to_csv(out_csv, index=False)

    out_png = out_figs / "pairwise_heatmap.png"
    _make_heatmap(
        res.table,
        phys=res.used_phys,
        bio=res.used_bio,
        out_png=out_png,
        title=f"{pid} | Pairwise Pearson phys×bio (min_frac={min_fraction}, min_n={min_points})",
    )

    log_lines = [
        f"INPUT: {in_path}",
        f"CONFIG: {cfg_path}",
        f"PLATFORM_ID: {pid}",
        "",
        f"min_fraction: {min_fraction}",
        f"min_points: {min_points}",
        "",
        f"phys_vars used ({len(res.used_phys)}): {res.used_phys}",
        f"bio_vars  used ({len(res.used_bio)}): {res.used_bio}",
        "",
        f"wrote: {out_csv}",
        f"wrote: {out_png}",
        "",
        "WARNINGS:",
    ] + (res.warnings if res.warnings else ["(none)"])

    _write_text(out_coup / "run_log.txt", log_lines)

    print(f"[OK] platform_id={pid}")
    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {out_png}")
    print(f"[OK] outputs in: {out_coup}")


if __name__ == "__main__":
    main()

"""
python -m bgcd.analysis.cli_coupling `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065378180/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots"

python -m bgcd.analysis.cli_coupling `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065379230/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots"

python -m bgcd.analysis.cli_coupling `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065470010/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots"
"""