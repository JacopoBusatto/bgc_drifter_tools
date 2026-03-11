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
def _scale_centered_01(x: float, x_min: float, x_mean: float, x_max: float) -> float:
    """
    Map:
      min  -> 0
      mean -> 0.5
      max  -> 1
    """
    if not np.isfinite(x) or not np.isfinite(x_min) or not np.isfinite(x_mean) or not np.isfinite(x_max):
        return np.nan

    if x_max <= x_min:
        return np.nan

    if x >= x_mean:
        denom = x_max - x_mean
        if denom == 0:
            return 0.5
        return 0.5 + 0.5 * (x - x_mean) / denom
    else:
        denom = x_mean - x_min
        if denom == 0:
            return 0.5
        return 0.5 * (x - x_min) / denom
    

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


def _compute_cluster_radar_profile(
    df_raw: pd.DataFrame,
    scores_seg: pd.DataFrame,
    *,
    vars_for_profile: List[str],
    time_col: str = "time_utc",
    rotation_var: str = "rotation_index",
) -> pd.DataFrame:
    """
    Radar profile in [0,1]:
      - standard vars: cluster mean mapped so that
            min -> 0, trajectory mean -> 0.5, max -> 1
      - rotation_var: cluster median mapped from [-1,1] to [0,1]
    """
    d = df_raw.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")

    s = scores_seg[[time_col, "cluster"]].copy()
    s[time_col] = pd.to_datetime(s[time_col], utc=True, errors="coerce")

    m = pd.merge(d, s, on=time_col, how="inner")

    vars_present = [v for v in vars_for_profile if v in m.columns]
    if not vars_present:
        return pd.DataFrame()

    for c in vars_present:
        m[c] = pd.to_numeric(m[c], errors="coerce")

    clusters = sorted(m["cluster"].dropna().unique())
    rows = []

    for cl in clusters:
        g = m.loc[m["cluster"] == cl].copy()
        row = {"cluster": int(cl)}

        for v in vars_present:
            gv = pd.to_numeric(g[v], errors="coerce")
            tv = pd.to_numeric(m[v], errors="coerce")

            if v == rotation_var:
                med = float(np.nanmedian(gv.to_numpy())) if gv.notna().any() else np.nan
                row[v] = 0.5 * med + 0.5 if np.isfinite(med) else np.nan
            else:
                x = float(np.nanmean(gv.to_numpy())) if gv.notna().any() else np.nan
                x_min = float(np.nanmin(tv.to_numpy())) if tv.notna().any() else np.nan
                x_mean = float(np.nanmean(tv.to_numpy())) if tv.notna().any() else np.nan
                x_max = float(np.nanmax(tv.to_numpy())) if tv.notna().any() else np.nan

                row[v] = _scale_centered_01(x, x_min, x_mean, x_max)

        rows.append(row)

    return pd.DataFrame(rows).set_index("cluster").sort_index()


def _smooth_time_series(
    df: pd.DataFrame,
    *,
    col: str,
    time_col: str = "time_utc",
    window: str = "12h",
    min_periods: int = 2,
) -> pd.Series:
    if col not in df.columns or time_col not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    d = df[[time_col, col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(subset=[time_col]).sort_values(time_col)

    s = (
        d.set_index(time_col)[col]
        .rolling(window=window, min_periods=min_periods)
        .mean()
    )

    out = pd.Series(index=d.index, data=s.to_numpy())
    out = out.reindex(df.index)

    return out

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
def _plot_best_lag_by_cluster(
    df_pair: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
) -> None:
    if df_pair.empty:
        return

    d = df_pair.copy()
    d = d.dropna(subset=["cluster", "best_lag_hours", "best_abs_r"])

    if d.empty:
        return

    fig = plt.figure(figsize=(8.8, 4.8))
    ax = fig.add_subplot(111)

    # jitter piccolo sull'asse x per non sovrapporre i punti
    rng = np.random.default_rng(0)
    x = d["cluster"].to_numpy(dtype=float)
    xj = x + rng.uniform(-0.08, 0.08, size=len(d))

    sizes = 40 + 220 * d["best_abs_r"].clip(lower=0, upper=1).to_numpy(dtype=float)

    sig = d["significant_any"].fillna(False).to_numpy(dtype=bool)

    # punti significativi
    if sig.any():
        ax.scatter(
            xj[sig],
            d.loc[sig, "best_lag_hours"],
            s=sizes[sig],
            alpha=0.85,
            label="significant",
        )

    # punti non significativi
    if (~sig).any():
        ax.scatter(
            xj[~sig],
            d.loc[~sig, "best_lag_hours"],
            s=sizes[~sig],
            alpha=0.35,
            marker="x",
            label="not significant",
        )

    # opzionale: etichetta con segment_id
    for _, row in d.iterrows():
        ax.annotate(
            f"{int(row['segment_id'])}",
            xy=(float(row["cluster"]), float(row["best_lag_hours"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    ax.axhline(0.0, linewidth=0.9, alpha=0.7)
    ax.set_xlabel("cluster")
    ax.set_ylabel("best lag (hours)")
    ax.set_title(title)

    clusts = sorted(pd.unique(d["cluster"]))
    ax.set_xticks(clusts)
    ax.set_xticklabels([f"{int(c)}" for c in clusts])

    ax.legend(loc="best", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


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
        # cmap="tab10",
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


def _compute_cluster_regime_profiles(
    df_raw: pd.DataFrame,
    scores_seg: pd.DataFrame,
    *,
    vars_for_profile: List[str],
    time_col: str = "time_utc",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - cluster_means_raw: mean per cluster in physical units
      - cluster_means_z:   z-score of cluster means using full-trajectory mean/std
      - cluster_means_mm:  min-max normalized cluster means using full trajectory min/max
    """
    d = df_raw.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")

    s = scores_seg[[time_col, "cluster"]].copy()
    s[time_col] = pd.to_datetime(s[time_col], utc=True, errors="coerce")

    m = pd.merge(d, s, on=time_col, how="inner")

    vars_present = [v for v in vars_for_profile if v in m.columns]
    if not vars_present:
        empty = pd.DataFrame()
        return empty, empty, empty

    for c in vars_present:
        m[c] = pd.to_numeric(m[c], errors="coerce")

    # raw cluster means
    cluster_means_raw = (
        m.groupby("cluster")[vars_present]
        .mean()
        .sort_index()
    )

    # trajectory-wide stats
    global_mean = m[vars_present].mean()
    global_std = m[vars_present].std(ddof=0).replace(0, np.nan)

    global_min = m[vars_present].min()
    global_max = m[vars_present].max()
    global_rng = (global_max - global_min).replace(0, np.nan)

    # z-score of cluster means
    cluster_means_z = (cluster_means_raw - global_mean) / global_std

    # min-max normalization of cluster means
    cluster_means_mm = (cluster_means_raw - global_min) / global_rng

    return cluster_means_raw, cluster_means_z, cluster_means_mm


def _plot_cluster_heatmap_zscore(
    df_z: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
) -> None:
    if df_z.empty:
        return

    fig = plt.figure(figsize=(1.2 * len(df_z.columns) + 2.5, 0.9 * len(df_z.index) + 2.2))
    ax = fig.add_subplot(111)

    arr = df_z.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(arr))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(df_z.columns)))
    ax.set_xticklabels(df_z.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(df_z.index)))
    ax.set_yticklabels([f"cluster {int(c)}" for c in df_z.index])

    # annotate cells
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("z-score of cluster mean\n(relative to full trajectory)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_cluster_radar_relative(
    df_radar: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
) -> None:
    if df_radar.empty or len(df_radar.columns) < 3:
        return

    labels = list(df_radar.columns)
    n = len(labels)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, polar=True)

    colors = plt.cm.tab10.colors

    for i, cl in enumerate(df_radar.index):
        vals = df_radar.loc[cl].to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        vals = vals.tolist()
        vals += vals[:1]

        ax.plot(
            angles,
            vals,
            linewidth=2,
            color=colors[i % len(colors)],
            label=f"cluster {int(cl)}",
        )
        ax.fill(
            angles,
            vals,
            alpha=0.10,
            color=colors[i % len(colors)],
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # dynamic radial limit
    # vmax = np.nanmax(df_radar.to_numpy(dtype=float))
    # if not np.isfinite(vmax):
    #     vmax = 1.5
    # vmax = max(1.2, min(vmax * 1.10, 3.0))
    vmax = 1

    ax.set_ylim(0, vmax)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_cluster_radar_minmax(
    df_mm: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
) -> None:
    if df_mm.empty or len(df_mm.columns) < 3:
        return

    labels = list(df_mm.columns)
    n = len(labels)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, polar=True)

    colors = plt.cm.tab10.colors

    for i, cl in enumerate(df_mm.index):
        vals = df_mm.loc[cl].to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        vals = vals.tolist()
        vals += vals[:1]

        ax.plot(angles, vals, linewidth=2, color=colors[i % len(colors)], label=f"cluster {int(cl)}")
        ax.fill(angles, vals, alpha=0.10, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=True)

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


def _plot_pc_space(
    scores_seg: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
) -> None:

    if "PC1" not in scores_seg.columns or "PC2" not in scores_seg.columns:
        return

    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111)

    clusters = scores_seg["cluster"].unique()
    colors = plt.cm.tab10.colors

    for cl in clusters:
        g = scores_seg.loc[scores_seg["cluster"] == cl]

        ax.scatter(
            g["PC1"],
            g["PC2"],
            s=35,
            alpha=0.8,
            label=f"cluster {cl}",
            color=colors[int(cl) % len(colors)],
        )

    ax.axhline(0, linewidth=0.8, alpha=0.4)
    ax.axvline(0, linewidth=0.8, alpha=0.4)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)

    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def _plot_ts_clusters(
    merged: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    smooth_window: str = "24h",
    smooth_min_periods: int = 3,
) -> None:

    if "salinity_psu" not in merged.columns:
        return

    if "sst_c" not in merged.columns and "temp_ctd_c" not in merged.columns:
        return

    T = "temp_ctd_c" if "temp_ctd_c" in merged.columns else "sst_c"

    d = merged.copy()
    d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True, errors="coerce")

    d = d.dropna(subset=["time_utc", "salinity_psu", T, "cluster"]).copy()
    if d.empty:
        return

    d = d.sort_values("time_utc").copy()

    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111)

    clusters = sorted(d["cluster"].unique())
    colors = plt.cm.tab10.colors

    # ------------------------------------------------------------------
    # scatter by cluster
    # ------------------------------------------------------------------
    for cl in clusters:
        g = d.loc[d["cluster"] == cl]

        ax.scatter(
            g["salinity_psu"],
            g[T],
            s=35,
            alpha=0.8,
            color=colors[int(cl) % len(colors)],
            label=f"cluster {cl}",
            edgecolor="none",
            zorder=2,
        )

    # ------------------------------------------------------------------
    # smoothed TS path in time order
    # ------------------------------------------------------------------
    ts_line = d[["time_utc", "salinity_psu", T]].copy()

    ts_smooth = (
        ts_line
        .rolling(
            smooth_window,
            on="time_utc",
            min_periods=smooth_min_periods,
        )[["salinity_psu", T]]
        .mean()
    )

    ts_smooth["time_utc"] = ts_line["time_utc"].values
    ts_smooth = ts_smooth.dropna(subset=["salinity_psu", T]).copy()

    if not ts_smooth.empty:
        ax.plot(
            ts_smooth["salinity_psu"],
            ts_smooth[T],
            color="black",
            linewidth=1.5,
            alpha=0.4,
            zorder=3,
            label=f"T-S path ({smooth_window} smooth)",
        )

        # start marker
        s0 = ts_smooth.iloc[0]
        ax.scatter(
            [s0["salinity_psu"]],
            [s0[T]],
            s=90,
            marker="o",
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
            zorder=4,
        )
        ax.annotate(
            "START",
            xy=(float(s0["salinity_psu"]), float(s0[T])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
        )

        # end marker
        s1 = ts_smooth.iloc[-1]
        ax.scatter(
            [s1["salinity_psu"]],
            [s1[T]],
            s=90,
            marker="s",
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
            zorder=4,
        )
        ax.annotate(
            "END",
            xy=(float(s1["salinity_psu"]), float(s1[T])),
            xytext=(6, -8),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="top",
        )

    ax.set_xlabel("Salinity (psu)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def _plot_lag_cluster(
    curves: list,
    out_png: Path,
    *,
    title: str,
):

    if not curves:
        return

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)

    colors = plt.cm.tab20.colors

    for i, item in enumerate(curves):

        df_lag = item["df_lag"]
        seg = item["segment_id"]

        color = colors[i % len(colors)]

        ax.plot(
            df_lag["lag_hours"],
            df_lag["r"],
            linewidth=1.4,
            color=color,
            label=f"seg {seg}",
        )

        sig = df_lag["p_value"].notna() & (df_lag["p_value"] < 0.05)

        if sig.any():

            ax.scatter(
                df_lag.loc[sig, "lag_hours"],
                df_lag.loc[sig, "r"],
                s=45,
                color=color,
                zorder=4,
            )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    ax.set_ylim(-1, 1)

    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("Pearson r")

    ax.set_title(title)

    ax.legend(ncol=2, fontsize=8)

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

    ap.add_argument("--rot-smooth", type=str, default=None, help="Optional time-based smoothing window for rotation_index before PCA, e.g. 12h")

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
    fig_lag_single_dir = plots_root / pid / "analysis" / "J_regime_lag" / "single"
    fig_lag_summary_dir = plots_root / pid / "analysis" / "J_regime_lag" / "summary"

    fig_profile_dir = plots_root / pid / "analysis" / "K_phys_regime_profiles"
    fig_bestlag_dir = plots_root / pid / "analysis" / "L_best_lag_by_cluster"

    _ensure_dir(fig_bestlag_dir)
    _ensure_dir(fig_profile_dir)
    _ensure_dir(out_lag_dir)
    _ensure_dir(out_lag_single_csv)
    _ensure_dir(fig_lag_single_dir)
    _ensure_dir(fig_lag_summary_dir)
    _ensure_dir(out_pca_dir)
    _ensure_dir(out_reg_dir)
    _ensure_dir(fig_pca_dir)
    _ensure_dir(fig_reg_dir)


    # -------------------------------------------------------------------------
    # optional smoothing of rotation index before PCA
    # -------------------------------------------------------------------------
    df_pca = df.copy()

    phys_vars_pca = list(phys_vars)

    if args.rot_smooth and "rotation_index" in df_pca.columns and "rotation_index" in phys_vars_pca:
        df_pca["rotation_index_smooth"] = _smooth_time_series(
            df_pca,
            col="rotation_index",
            time_col="time_utc",
            window=str(args.rot_smooth),
            min_periods=2,
        )

        phys_vars_pca = [
            "rotation_index_smooth" if v == "rotation_index" else v
            for v in phys_vars_pca
        ]
    else:
        phys_vars_pca = phys_vars
    # -------------------------------------------------------------------------
    # physical PCA only
    # -------------------------------------------------------------------------
    keep_cols = [c for c in ["lat", "lon"] if c in df.columns]

    res = run_pca(
        df_pca,
        variables=phys_vars_pca,
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


    _plot_ts_clusters(
        merged,
        fig_reg_dir / "ts_clusters.png",
        title=f"{pid} | TS diagram colored by cluster",
        smooth_window="36h",
    )

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

    profile_vars_phys = [
        v for v in phys_vars
        if v in merged.columns
    ]

    profile_vars_bio = [
        v for v in ["chl", "bbp_532_m1", "DO2_c"]
        if v in merged.columns
    ]
    # -------------------------------------------------------------------------
    # cluster regime profiles
    # -------------------------------------------------------------------------
    prof_raw_phys, prof_z_phys, prof_mm_phys = _compute_cluster_regime_profiles(
        df,
        scores_seg,
        vars_for_profile=profile_vars_phys,
        time_col="time_utc",
    )

    if not prof_raw_phys.empty:
        prof_raw_phys.to_csv(out_reg_dir / "cluster_profile_phys_raw.csv")
        prof_z_phys.to_csv(out_reg_dir / "cluster_profile_phys_zscore.csv")
        prof_mm_phys.to_csv(out_reg_dir / "cluster_profile_phys_minmax.csv")

        _plot_cluster_heatmap_zscore(
            prof_z_phys,
            fig_profile_dir / "cluster_profile_phys_heatmap_zscore.png",
            title=f"{pid} | physical regime profile (z-score heatmap)",
        )

        prof_radar_phys = _compute_cluster_radar_profile(
            df,
            scores_seg,
            vars_for_profile=profile_vars_phys,
            time_col="time_utc",
            rotation_var="rotation_index",
        )

        if not prof_radar_phys.empty:
            prof_radar_phys.to_csv(out_reg_dir / "cluster_profile_phys_radar.csv")

            prof_radar_phys_plot = prof_radar_phys#.rename(
            #     columns={"rotation_index": "RI_median→[0,1]"}
            # )

            _plot_cluster_radar_relative(
                prof_radar_phys_plot,
                fig_profile_dir / "cluster_profile_phys_radar_relative.png",
                title=f"{pid} | physical regime profile (radar, cluster/traj mean; RI median mapped to [0,1])",
            )
        prof_raw_bio, prof_z_bio, prof_mm_bio = _compute_cluster_regime_profiles(
        df,
        scores_seg,
        vars_for_profile=profile_vars_bio,
        time_col="time_utc",
    )

    if not prof_raw_bio.empty:
        prof_raw_bio.to_csv(out_reg_dir / "cluster_profile_bio_raw.csv")
        prof_z_bio.to_csv(out_reg_dir / "cluster_profile_bio_zscore.csv")
        prof_mm_bio.to_csv(out_reg_dir / "cluster_profile_bio_minmax.csv")

        _plot_cluster_heatmap_zscore(
            prof_z_bio,
            fig_profile_dir / "cluster_profile_bio_heatmap_zscore.png",
            title=f"{pid} | biogeochemical regime signature (z-score heatmap)",
        )

        prof_radar_bio = _compute_cluster_radar_profile(
            df,
            scores_seg,
            vars_for_profile=profile_vars_bio,
            time_col="time_utc",
            rotation_var="rotation_index",  # irrilevante qui, ma ok lasciarlo
        )

        if not prof_radar_bio.empty:
            prof_radar_bio.to_csv(out_reg_dir / "cluster_profile_bio_radar.csv")

            _plot_cluster_radar_relative(
                prof_radar_bio,
                fig_profile_dir / "cluster_profile_bio_radar_relative.png",
                title=f"{pid} | biogeochemical regime signature (radar, min→0, mean→0.5, max→1)",
            )
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

    _plot_pc_space(
        scores_seg,
        fig_reg_dir / "pc_space_clusters.png",
        title=f"{pid} | PC space (PC1 vs PC2)",
    )

    # -------------------------------------------------------------------------
    # lag correlation by valid segment
    # -------------------------------------------------------------------------
    pairs = _make_variable_pairs(phys_vars, bio_vars, merged.columns)

    lag_summary_rows: List[Dict[str, Any]] = []
    summary_curves: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

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
                out_png = fig_lag_single_dir / f"lag_seg{seg_id:03d}_cl{cluster_id}_{pair_name}.png"
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
                    "rot_smooth": args.rot_smooth,
                    "phys_vars_pca": phys_vars_pca,
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

            summary_curves.setdefault(pair_name, {})
            summary_curves[pair_name].setdefault(cluster_id, [])

            summary_curves[pair_name][cluster_id].append(
                {
                    "df_lag": res_lag.df.copy(),
                    "segment_id": seg_id,
                }
            )


    lag_summary_df = pd.DataFrame(lag_summary_rows)
    lag_summary_df.to_csv(out_lag_dir / "regime_lag_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # best lag by cluster (one plot per variable pair)
    # -------------------------------------------------------------------------
    if not lag_summary_df.empty:
        for pair_name, gpair in lag_summary_df.groupby("pair_name", sort=True):
            out_png = fig_bestlag_dir / f"best_lag_{pair_name}.png"
            _plot_best_lag_by_cluster(
                gpair,
                out_png,
                title=f"{pid} | best lag by cluster | {pair_name}",
            )

    if write_summary_lag_plots:
        for pair_name, cluster_dict in summary_curves.items():

            pair_dir = fig_lag_summary_dir / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)

            for cluster_id, curves in cluster_dict.items():
                out_png = pair_dir / f"lag_cluster{cluster_id}.png"

                _plot_lag_cluster(
                    curves,
                    out_png,
                    title=f"{pid} | {pair_name} | cluster {cluster_id}",
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
        f"  - {fig_lag_single_dir}",
        f"  - {fig_lag_summary_dir}",
        "outputs:",
        f"  - {out_pca_dir / 'pca_loadings.csv'}",
        f"  - {out_pca_dir / 'pca_scores.csv'}",
        f"  - {out_pca_dir / 'pca_explained_variance.csv'}",
        f"  - {out_reg_dir / 'pca_clusters_segments.csv'}",
        f"  - {out_reg_dir / 'regime_segments.csv'}",
        f"  - {out_reg_dir / 'cluster_summary.csv'}",
        f"  - {fig_reg_dir / 'trajectory_clusters.png'}",
        f"  - {out_reg_dir / 'cluster_profile_phys_raw.csv'}",
        f"  - {out_reg_dir / 'cluster_profile_phys_zscore.csv'}",
        f"  - {out_reg_dir / 'cluster_profile_phys_minmax.csv'}",
        f"  - {fig_profile_dir / 'cluster_profile_phys_heatmap_zscore.png'}",
        f"  - {fig_profile_dir / 'cluster_profile_phys_radar_minmax.png'}",
        f"  - {fig_profile_dir / 'cluster_profile_bio_heatmap_zscore.png'}",
        f"  - {fig_profile_dir / 'cluster_profile_bio_radar_minmax.png'}",
        f"  - {out_reg_dir / 'cluster_profile_phys_radar.csv'}",
        f"  - {fig_profile_dir / 'cluster_profile_phys_radar_relative.png'}",
        f"  - {fig_bestlag_dir}",
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
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065378180/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min_no_sal.yml" `
  --out-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --rot-smooth 24h `
  --k 2 `
  --use-pcs 1 2 3 `
  --segment-min-duration-days 1.0 `
  --segment-min-points 8 `
  --merge-short-segments-points 9 `
  --min-pairs 10 `
  --write-single-lag-plots false `
  --write-summary-lag-plots true


python -m bgcd.analysis.cli_phys_regime_lag `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065379230/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --out-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --rot-smooth 24h `
  --k 4 `
  --use-pcs 1 2 3 4 `
  --segment-min-duration-days 1.0 `
  --segment-min-points 8 `
  --merge-short-segments-points 9 `
  --min-pairs 10 `
  --write-single-lag-plots false `
  --write-summary-lag-plots true

python -m bgcd.analysis.cli_phys_regime_lag `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065470010/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --out-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --rot-smooth 24h `
  --k 4 `
  --use-pcs 1 2 3 4 `
  --segment-min-duration-days 1.0 `
  --segment-min-points 8 `
  --merge-short-segments-points 9 `
  --min-pairs 10 `
  --write-single-lag-plots false `
  --write-summary-lag-plots true
"""