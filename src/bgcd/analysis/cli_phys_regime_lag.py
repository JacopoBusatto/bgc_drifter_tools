from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from bgcd.analysis.pca import run_pca
from bgcd.analysis.cli_window import (
    _load_yaml,
    _dedup_keep_order,
    _platform_id_from_df_or_filename,
)
from bgcd.analysis.lag_corr import lag_correlation_asof

# -----------------------------------------------------------------------------
# small utils
# -----------------------------------------------------------------------------
def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y", "on")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_use_pcs(vals: List[int]) -> List[str]:
    pcs: List[str] = []
    for v in vals:
        if int(v) <= 0:
            raise ValueError("--use-pcs must contain positive integers (1-based)")
        pcs.append(f"PC{int(v)}")
    return pcs


# -----------------------------------------------------------------------------
# segment logic
# -----------------------------------------------------------------------------
def _build_segments(
    scores: pd.DataFrame,
    *,
    time_col: str = "time_utc",
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """
    Build contiguous cluster segments in time order.

    A new segment starts when:
      - cluster changes
      - or time is missing / cluster missing
    """
    d = scores.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col, cluster_col]).sort_values(time_col).reset_index(drop=True)

    if d.empty:
        d["segment_id"] = pd.Series(dtype="int64")
        return d

    cluster_change = d[cluster_col].ne(d[cluster_col].shift(1))
    d["segment_id"] = cluster_change.cumsum() - 1
    d["segment_id"] = d["segment_id"].astype(int)

    return d


def _merge_short_segments(
    scores_seg: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    segment_col: str = "segment_id",
    min_run_points: int = 3,
) -> pd.DataFrame:
    """
    Merge very short segments into neighbors when possible.

    Rules:
      1) if segment length < min_run_points and previous and next segments
         exist and have the same cluster, reassign to that cluster.
      2) otherwise, if previous and next segments exist but have different
         clusters, reassign to the longer neighbor.
         If equal length, assign to previous.
    """
    d = scores_seg.copy()

    if d.empty:
        return d

    changed = True

    # Iterate until stable, because one merge can enable another
    while changed:
        changed = False

        seg_sizes = d.groupby(segment_col).size().to_dict()
        seg_ids = sorted(d[segment_col].unique())

        for i, seg in enumerate(seg_ids):
            seg_len = int(seg_sizes.get(seg, 0))
            if seg_len >= int(min_run_points):
                continue

            prev_seg = seg_ids[i - 1] if i > 0 else None
            next_seg = seg_ids[i + 1] if i < len(seg_ids) - 1 else None

            if prev_seg is None and next_seg is None:
                continue

            # edge segments: merge into the only available neighbor
            if prev_seg is None:
                next_cluster = d.loc[d[segment_col] == next_seg, cluster_col].iloc[0]
                d.loc[d[segment_col] == seg, cluster_col] = next_cluster
                changed = True
                break

            if next_seg is None:
                prev_cluster = d.loc[d[segment_col] == prev_seg, cluster_col].iloc[0]
                d.loc[d[segment_col] == seg, cluster_col] = prev_cluster
                changed = True
                break

            prev_cluster = d.loc[d[segment_col] == prev_seg, cluster_col].iloc[0]
            next_cluster = d.loc[d[segment_col] == next_seg, cluster_col].iloc[0]
            prev_len = int(seg_sizes.get(prev_seg, 0))
            next_len = int(seg_sizes.get(next_seg, 0))

            if prev_cluster == next_cluster:
                d.loc[d[segment_col] == seg, cluster_col] = prev_cluster
                changed = True
                break

            # lonely short segment between two different clusters:
            # absorb into the longer neighbor
            if prev_len >= next_len:
                d.loc[d[segment_col] == seg, cluster_col] = prev_cluster
            else:
                d.loc[d[segment_col] == seg, cluster_col] = next_cluster

            changed = True
            break

        if changed:
            d = _build_segments(d, cluster_col=cluster_col)

    return d

def _summarize_segments(
    scores_seg: pd.DataFrame,
    *,
    time_col: str = "time_utc",
    cluster_col: str = "cluster",
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if scores_seg.empty:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "cluster",
                "start",
                "end",
                "duration_h",
                "duration_days",
                "n_points",
            ]
        )

    d = scores_seg.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")

    for seg_id, g in d.groupby(segment_col, sort=True):
        g = g.sort_values(time_col)
        t0 = g[time_col].iloc[0]
        t1 = g[time_col].iloc[-1]
        dur_h = (t1 - t0).total_seconds() / 3600.0 if pd.notna(t0) and pd.notna(t1) else np.nan

        rows.append(
            {
                "segment_id": int(seg_id),
                "cluster": int(g[cluster_col].mode().iloc[0]),
                "start": t0,
                "end": t1,
                "duration_h": dur_h,
                "duration_days": dur_h / 24.0 if pd.notna(dur_h) else np.nan,
                "n_points": int(len(g)),
            }
        )

    return pd.DataFrame(rows).sort_values(["start", "segment_id"]).reset_index(drop=True)


def _flag_valid_segments(
    seg_summary: pd.DataFrame,
    *,
    min_duration_days: float,
    min_points: int,
) -> pd.DataFrame:
    d = seg_summary.copy()

    if d.empty:
        d["is_valid"] = pd.Series(dtype=bool)
        return d

    d["is_valid"] = (
        (pd.to_numeric(d["duration_days"], errors="coerce") >= float(min_duration_days))
        & (pd.to_numeric(d["n_points"], errors="coerce") >= int(min_points))
    )
    return d


# -----------------------------------------------------------------------------
# lag correlation logic
# -----------------------------------------------------------------------------
def _make_variable_pairs(phys_vars: List[str], bio_vars: List[str], df_cols: Iterable[str]) -> List[Tuple[str, str]]:
    """
    Build all unordered pairs from phys_vars + bio_vars, preserving order:
      phys first, then bio.
    """
    vars_all = _dedup_keep_order(list(phys_vars) + list(bio_vars))
    vars_all = [v for v in vars_all if v in set(df_cols)]

    pairs: List[Tuple[str, str]] = []
    for i in range(len(vars_all)):
        for j in range(i + 1, len(vars_all)):
            pairs.append((vars_all[i], vars_all[j]))
    return pairs


def _best_lag_row(df_lag: pd.DataFrame) -> pd.Series | None:
    d = df_lag.dropna(subset=["r"]).copy()
    if d.empty:
        return None
    d["abs_r"] = d["r"].abs()
    return d.sort_values("abs_r", ascending=False).iloc[0]


# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------
def _plot_trajectory_clusters(
    scores_seg: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
) -> None:
    if "lat" not in scores_seg.columns or "lon" not in scores_seg.columns:
        return
    if "cluster" not in scores_seg.columns:
        return

    d = scores_seg.dropna(subset=["lat", "lon", "cluster"]).copy()
    if d.empty:
        return

    fig = plt.figure(figsize=(7.8, 6.5))
    ax = fig.add_subplot(111)

    sc = ax.scatter(
        d["lon"],
        d["lat"],
        c=d["cluster"],
        s=30,
        alpha=0.9,
        edgecolor="none",
        cmap="tab10",
    )

    # thin path underneath
    ax.plot(d["lon"], d["lat"], linewidth=0.8, alpha=0.35)

    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(title)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("cluster")

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_lag_curve_single(
    df_lag: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
) -> None:
    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(111)

    d = df_lag.copy()
    ax.plot(d["lag_hours"], d["r"], linewidth=1.5)

    sig = d["p_value"].notna() & (d["p_value"] < 0.05)
    if sig.any():
        ax.scatter(
            d.loc[sig, "lag_hours"],
            d.loc[sig, "r"],
            s=55,
            marker="o",
            zorder=4,
            label="p < 0.05",
        )

    ax.axhline(0.0, linewidth=0.9, alpha=0.7)
    ax.axvline(0.0, linewidth=0.9, alpha=0.5)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("lag (hours)  [positive => y after x]")
    ax.set_ylabel("Pearson r")
    ax.set_title(title)

    best = _best_lag_row(d)
    if best is not None:
        ax.axvline(float(best["lag_hours"]), linewidth=0.9, alpha=0.5)
        ax.annotate(
            f"best: lag={int(best['lag_hours'])}h, r={best['r']:.2f}, n={int(best['n_pairs'])}",
            xy=(float(best["lag_hours"]), float(best["r"])),
            xytext=(10, 20),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    if sig.any():
        ax.legend(loc="best", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_lag_curve_summary(
    curves: List[Dict[str, Any]],
    out_png: Path,
    *,
    title: str,
) -> None:
    """
    curves: list of dicts with keys:
      - df_lag
      - cluster
      - segment_id
    """
    if not curves:
        return

    fig = plt.figure(figsize=(11.2, 5.2))
    ax = fig.add_subplot(111)

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for item in curves:
        df_lag = item["df_lag"]
        cluster = int(item["cluster"])
        segment_id = int(item["segment_id"])

        color = colors[cluster % len(colors)]
        marker = markers[segment_id % len(markers)]

        ax.plot(
            df_lag["lag_hours"],
            df_lag["r"],
            linewidth=1.3,
            alpha=0.9,
            color=color,
            label=f"cl{cluster}-seg{segment_id}",
        )

        sig = df_lag["p_value"].notna() & (df_lag["p_value"] < 0.05)
        if sig.any():
            ax.scatter(
                df_lag.loc[sig, "lag_hours"],
                df_lag.loc[sig, "r"],
                s=42,
                marker=marker,
                color=color,
                zorder=4,
            )

    ax.axhline(0.0, linewidth=0.9, alpha=0.7)
    ax.axvline(0.0, linewidth=0.9, alpha=0.5)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("lag (hours)  [positive => y after x]")
    ax.set_ylabel("Pearson r")
    ax.set_title(title)

    # avoid giant legend if many curves
    if len(curves) <= 18:
        ax.legend(loc="best", frameon=True, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bgcd.analysis.cli_phys_regime_lag",
        description=(
            "Step 1: physical PCA + clustering + contiguous regime segmentation. "
            "Writes dedicated phys-regime outputs without touching the existing phys+bio PCA outputs."
        ),
    )
    ap.add_argument("--input", required=True, help="CSV to analyze (recommended: best-window subset).")
    ap.add_argument("--config", required=True, help="analysis_config_min.yml")
    ap.add_argument("--out-root", required=True, help="Root OUT directory")
    ap.add_argument("--plots-root", required=True, help="Root plots directory")

    ap.add_argument("--n-components", type=int, default=None, help="Optional PCA n_components")
    ap.add_argument("--k", type=int, default=3, help="Number of KMeans clusters")
    ap.add_argument("--use-pcs", type=int, nargs="+", default=[1, 2], help="PC indices used for clustering")
    ap.add_argument("--random-state", type=int, default=0)

    ap.add_argument("--segment-min-duration-days", type=float, default=1.0, help="Minimum segment duration to keep")
    ap.add_argument("--segment-min-points", type=int, default=8, help="Minimum number of points to keep a segment")
    ap.add_argument("--merge-short-segments-points", type=int, default=3, help="Merge segments shorter than this number of points")

    ap.add_argument("--max-lag-hours", type=int, default=None, help="Override YAML max_lag_hours")
    ap.add_argument("--lag-step-hours", type=int, default=None, help="Override YAML lag_step_hours")
    ap.add_argument("--tolerance-hours", type=float, default=None, help="Override YAML lag_match_tolerance_hours")
    ap.add_argument("--min-pairs", type=int, default=10, help="Min matched pairs to compute r at a lag")
    ap.add_argument("--write-single-lag-plots", type=str, default="true", help="true/false")
    ap.add_argument("--write-summary-lag-plots", type=str, default="true", help="true/false")
    args = ap.parse_args()

    in_path = Path(args.input)
    cfg_path = Path(args.config)
    out_root = Path(args.out_root)
    plots_root = Path(args.plots_root)

    # -------------------------------------------------------------------------
    # load input + config
    # -------------------------------------------------------------------------
    df = pd.read_csv(in_path)
    pid = _platform_id_from_df_or_filename(df, in_path)

    cfg = _load_yaml(cfg_path)
    analysis_cfg: Dict[str, Any] = (cfg or {}).get("analysis", {}) or {}

    coupling = (analysis_cfg.get("coupling", {}) or {})

    max_lag = int(args.max_lag_hours) if args.max_lag_hours is not None else int(coupling.get("max_lag_hours", 120))
    step = int(args.lag_step_hours) if args.lag_step_hours is not None else int(float(coupling.get("lag_step_hours", 3)))
    tol = float(args.tolerance_hours) if args.tolerance_hours is not None else float(coupling.get("lag_match_tolerance_hours", 1.5))

    write_single_lag_plots = _as_bool(args.write_single_lag_plots) if "_as_bool" in globals() else str(args.write_single_lag_plots).strip().lower() in ("1", "true", "yes", "y", "on")
    write_summary_lag_plots = _as_bool(args.write_summary_lag_plots) if "_as_bool" in globals() else str(args.write_summary_lag_plots).strip().lower() in ("1", "true", "yes", "y", "on")

    phys_vars = _dedup_keep_order(analysis_cfg.get("phys_vars", []) or [])
    bio_vars = _dedup_keep_order(analysis_cfg.get("bio_vars", []) or [])
    all_vars = _dedup_keep_order(list(phys_vars) + list(bio_vars))

    if not phys_vars:
        raise SystemExit("[ERR] No phys_vars found in config.")

    # -------------------------------------------------------------------------
    # outputs
    # -------------------------------------------------------------------------
    out_pca_dir = out_root / pid / "H_phys_pca"
    out_reg_dir = out_root / pid / "I_phys_regimes"

    fig_pca_dir = plots_root / pid / "analysis" / "H_phys_pca"
    fig_reg_dir = plots_root / pid / "analysis" / "I_phys_regimes"

    out_lag_dir = out_root / pid / "J_regime_lag"
    out_lag_single_csv = out_lag_dir / "csv_single"
    out_lag_single_fig = plots_root / pid / "analysis" / "J_regime_lag" / "single"
    out_lag_summary_fig = plots_root / pid / "analysis" / "J_regime_lag" / "summary"

    _ensure_dir(out_lag_dir)
    _ensure_dir(out_lag_single_csv)
    _ensure_dir(out_lag_single_fig)
    _ensure_dir(out_lag_summary_fig)
    _ensure_dir(out_pca_dir)
    _ensure_dir(out_reg_dir)
    _ensure_dir(fig_pca_dir)
    _ensure_dir(fig_reg_dir)

    # -------------------------------------------------------------------------
    # physical PCA only
    # -------------------------------------------------------------------------
    keep_cols = [c for c in ["lat", "lon"] if c in df.columns]

    res = run_pca(
        df,
        variables=phys_vars,
        time_col="time_utc",
        keep_cols=keep_cols,
        n_components=args.n_components,
    )

    res.loadings.to_csv(out_pca_dir / "pca_loadings.csv")
    res.scores.to_csv(out_pca_dir / "pca_scores.csv", index=False)
    res.explained.to_csv(out_pca_dir / "pca_explained_variance.csv", index=False)

    # -------------------------------------------------------------------------
    # clustering on selected PCs
    # -------------------------------------------------------------------------
    scores = res.scores.copy()
    scores["time_utc"] = pd.to_datetime(scores["time_utc"], utc=True, errors="coerce")
    scores = scores.dropna(subset=["time_utc"]).copy()

    pcs = _parse_use_pcs(args.use_pcs)
    for pc in pcs:
        if pc not in scores.columns:
            raise SystemExit(f"[ERR] {pc} not found in PCA scores columns.")

    X = scores[pcs].to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1)
    scores = scores.loc[ok].copy()
    X = X[ok]

    if len(scores) < max(5, args.k):
        raise SystemExit(f"[ERR] Too few PCA-score rows for clustering: {len(scores)}")

    km = KMeans(
        n_clusters=int(args.k),
        random_state=int(args.random_state),
        n_init="auto",
    )
    
    scores["cluster"] = km.fit_predict(X)
    scores["cluster_raw"] = scores["cluster"]
    # -------------------------------------------------------------------------
    # segments
    # -------------------------------------------------------------------------
    scores_seg = _build_segments(scores, time_col="time_utc", cluster_col="cluster")

    # merge short segments
    scores_seg = _merge_short_segments(
        scores_seg,
        cluster_col="cluster",
        segment_col="segment_id",
        min_run_points=int(args.merge_short_segments_points),
    )

    seg_summary = _summarize_segments(scores_seg, time_col="time_utc", cluster_col="cluster", segment_col="segment_id")
    seg_summary = _flag_valid_segments(
        seg_summary,
        min_duration_days=float(args.segment_min_duration_days),
        min_points=int(args.segment_min_points),
    )

    valid_seg_ids = set(seg_summary.loc[seg_summary["is_valid"], "segment_id"].astype(int).tolist())
    scores_seg["segment_is_valid"] = scores_seg["segment_id"].isin(valid_seg_ids)

    # -------------------------------------------------------------------------
    # cluster summary on original variables
    # -------------------------------------------------------------------------
    d_raw = df.copy()
    d_raw["time_utc"] = pd.to_datetime(d_raw["time_utc"], utc=True, errors="coerce")

    s_join = scores_seg[["time_utc", "cluster", "segment_id", "segment_is_valid"]].copy()
    merged = pd.merge(d_raw, s_join, on="time_utc", how="inner")

    rows_summary: List[Dict[str, Any]] = []
    vars_for_summary = [c for c in all_vars if c in merged.columns]

    for cl in sorted(merged["cluster"].dropna().unique()):
        g = merged.loc[merged["cluster"] == cl].copy()
        row: Dict[str, Any] = {"cluster": int(cl), "n": int(len(g))}
        for c in vars_for_summary:
            v = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}__median"] = float(np.nanmedian(v.to_numpy())) if v.notna().any() else np.nan
            row[f"{c}__mean"] = float(np.nanmean(v.to_numpy())) if v.notna().any() else np.nan
            row[f"{c}__std"] = float(np.nanstd(v.to_numpy())) if v.notna().any() else np.nan
        rows_summary.append(row)

    cluster_summary = pd.DataFrame(rows_summary)

    # -------------------------------------------------------------------------
    # write outputs
    # -------------------------------------------------------------------------
    scores_seg.to_csv(out_reg_dir / "pca_clusters_segments.csv", index=False)
    seg_summary.to_csv(out_reg_dir / "regime_segments.csv", index=False)
    cluster_summary.to_csv(out_reg_dir / "cluster_summary.csv", index=False)

    _plot_trajectory_clusters(
        scores_seg,
        fig_reg_dir / "trajectory_clusters.png",
        title=f"{pid} | trajectory colored by cluster (phys PCA, k={args.k})",
    )


        # -------------------------------------------------------------------------
    # lag correlation by valid segment
    # -------------------------------------------------------------------------
    pairs = _make_variable_pairs(phys_vars, bio_vars, merged.columns)

    lag_summary_rows: List[Dict[str, Any]] = []
    summary_curves: Dict[str, List[Dict[str, Any]]] = {}

    valid_seg_ids_sorted = (
        seg_summary.loc[seg_summary["is_valid"], "segment_id"]
        .astype(int)
        .sort_values()
        .tolist()
    )

    for seg_id in valid_seg_ids_sorted:
        seg_meta = seg_summary.loc[seg_summary["segment_id"] == seg_id].iloc[0]
        cluster_id = int(seg_meta["cluster"])

        g = merged.loc[merged["segment_id"] == seg_id].copy()
        g = g.sort_values("time_utc")

        if g.empty:
            continue

        for x, y in pairs:
            if x not in g.columns or y not in g.columns:
                continue

            try:
                res_lag = lag_correlation_asof(
                    g,
                    x=x,
                    y=y,
                    time_col="time_utc",
                    max_lag_hours=max_lag,
                    lag_step_hours=step,
                    match_tolerance_hours=tol,
                    min_pairs=int(args.min_pairs),
                )
            except Exception:
                continue

            pair_name = f"{x}__{y}"

            out_csv = out_lag_single_csv / f"lag_seg{seg_id:03d}_cl{cluster_id}_{pair_name}.csv"
            res_lag.df.to_csv(out_csv, index=False)

            if write_single_lag_plots:
                out_png = out_lag_single_fig / f"lag_seg{seg_id:03d}_cl{cluster_id}_{pair_name}.png"
                _plot_lag_curve_single(
                    res_lag.df,
                    out_png,
                    title=f"{pid} | seg {seg_id} cl {cluster_id} | {pair_name}",
                )

            best = _best_lag_row(res_lag.df)

            sig = res_lag.df["p_value"].notna() & (res_lag.df["p_value"] < 0.05)
            n_sig = int(sig.sum())

            lag_summary_rows.append(
                {
                    "segment_id": int(seg_id),
                    "cluster": int(cluster_id),
                    "start": seg_meta["start"],
                    "end": seg_meta["end"],
                    "duration_h": float(seg_meta["duration_h"]),
                    "duration_days": float(seg_meta["duration_days"]),
                    "n_points": int(seg_meta["n_points"]),
                    "x": x,
                    "y": y,
                    "pair_name": pair_name,
                    "best_lag_hours": float(best["lag_hours"]) if best is not None else np.nan,
                    "best_r": float(best["r"]) if best is not None else np.nan,
                    "best_abs_r": float(abs(best["r"])) if best is not None else np.nan,
                    "best_p_value": float(best["p_value"]) if best is not None and pd.notna(best["p_value"]) else np.nan,
                    "best_n_pairs": int(best["n_pairs"]) if best is not None and pd.notna(best["n_pairs"]) else np.nan,
                    "n_significant_lags": n_sig,
                    "significant_any": bool(n_sig > 0),
                    "csv_path": str(out_csv),
                }
            )

            summary_curves.setdefault(pair_name, []).append(
                {
                    "df_lag": res_lag.df.copy(),
                    "cluster": cluster_id,
                    "segment_id": seg_id,
                }
            )

    lag_summary_df = pd.DataFrame(lag_summary_rows)
    lag_summary_df.to_csv(out_lag_dir / "regime_lag_summary.csv", index=False)

    if write_summary_lag_plots:
        for pair_name, curves in summary_curves.items():
            out_png = out_lag_summary_fig / f"lag_summary_{pair_name}.png"
            _plot_lag_curve_summary(
                curves,
                out_png,
                title=f"{pid} | lag summary | {pair_name}",
            )

    run_log = [
        f"INPUT: {in_path}",
        f"CONFIG: {cfg_path}",
        f"PLATFORM_ID: {pid}",
        "",
        f"phys_vars: {phys_vars}",
        f"bio_vars: {bio_vars}",
        "",
        f"n_rows_in: {res.n_rows_in}",
        f"n_rows_used_pca: {res.n_rows_used}",
        f"k: {args.k}",
        f"use_pcs: {pcs}",
        f"random_state: {args.random_state}",
        f"segment_min_duration_days: {args.segment_min_duration_days}",
        f"segment_min_points: {args.segment_min_points}",
        f"merge_short_segments_points: {args.merge_short_segments_points}",
        "",
        f"n_segments_total: {len(seg_summary)}",
        f"n_segments_valid: {int(seg_summary['is_valid'].sum()) if 'is_valid' in seg_summary.columns else 0}",
        "",
        f"max_lag_hours: {max_lag}",
        f"lag_step_hours: {step}",
        f"lag_match_tolerance_hours: {tol}",
        f"min_pairs: {args.min_pairs}",
        f"n_pairs_total: {len(pairs)}",
        f"n_valid_segments: {len(valid_seg_ids_sorted)}",
        f"n_lag_results: {len(lag_summary_df)}",
        "",
        f"  - {out_lag_dir / 'regime_lag_summary.csv'}",
        f"  - {out_lag_single_csv}",
        f"  - {out_lag_single_fig}",
        f"  - {out_lag_summary_fig}",
        "outputs:",
        f"  - {out_pca_dir / 'pca_loadings.csv'}",
        f"  - {out_pca_dir / 'pca_scores.csv'}",
        f"  - {out_pca_dir / 'pca_explained_variance.csv'}",
        f"  - {out_reg_dir / 'pca_clusters_segments.csv'}",
        f"  - {out_reg_dir / 'regime_segments.csv'}",
        f"  - {out_reg_dir / 'cluster_summary.csv'}",
        f"  - {fig_reg_dir / 'trajectory_clusters.png'}",
    ]
    _write_text(out_reg_dir / "run_log.txt", run_log)

    print(f"[OK] platform_id={pid}")
    print(f"[OK] phys PCA outputs: {out_pca_dir}")
    print(f"[OK] regime outputs:   {out_reg_dir}")
    print(f"[OK] valid segments:   {int(seg_summary['is_valid'].sum()) if 'is_valid' in seg_summary.columns else 0}")


if __name__ == "__main__":
    main()



"""
python -m bgcd.analysis.cli_phys_regime_lag `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065379230/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --out-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --k 3 `
  --use-pcs 1 2 3 `
  --segment-min-duration-days 1.0 `
  --segment-min-points 8 `
  --merge-short-segments-points 4 `
  --min-pairs 10 `
  --write-single-lag-plots true `
  --write-summary-lag-plots true
"""