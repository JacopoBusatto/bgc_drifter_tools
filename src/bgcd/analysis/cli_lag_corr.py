# ================================
# File: src/bgcd/analysis/cli_lag_corr.py
# ================================
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from bgcd.analysis.lag_corr import lag_correlation_asof
from bgcd.analysis.cli_window import _load_yaml, _platform_id_from_df_or_filename


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_pairs_arg(pairs_str: str) -> List[Tuple[str, str]]:
    """
    Format: "x1:y1,x2:y2"
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


def _plot_lag_curve(df_lag: pd.DataFrame, out_png: Path, title: str) -> None:
    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(df_lag["lag_hours"], df_lag["r"])
    ax.axhline(0.0, linewidth=0.9)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("lag (hours)  [positive => y after x]")
    ax.set_ylabel("Pearson r")
    ax.set_title(title)

    # annotate best lag (by abs(r), among valid points)
    d = df_lag.dropna(subset=["r"]).copy()

    # mark significant lags (p < 0.05)
    if "p_value" in d.columns:
        sig = d["p_value"].notna() & (d["p_value"] < 0.05)

        if sig.any():
            ax.scatter(
                d.loc[sig, "lag_hours"],
                d.loc[sig, "r"],
                s=60,
                marker="*",
                zorder=4,
                label="p < 0.05",
            )

    if not d.empty:
        d["abs_r"] = d["r"].abs()
        best = d.sort_values("abs_r", ascending=False).iloc[0]
        ax.axvline(best["lag_hours"], linewidth=0.9, alpha=0.6)
        ax.annotate(
            f"best: lag={int(best['lag_hours'])}h, r={best['r']:.2f}, n={int(best['n_pairs'])}",
            xy=(float(best["lag_hours"]), float(best["r"])),
            xytext=(10, 20),            # spostamento (x,y) in punti
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bgcd.analysis.cli_lag_corr",
        description="Lagged correlation corr(x(t), y(t+lag)) without interpolation using merge_asof.",
    )
    ap.add_argument("--input", required=True, help="CSV to analyze (recommended: best-window subset).")
    ap.add_argument("--config", required=True, help="analysis_config_min.yml")
    ap.add_argument("--outdir", default="OUT", help="Base output directory. Outputs are isolated in OUT/<platform_id>/analysis/lag/...")
    ap.add_argument("--plots-root", default="plots", help="Root folder for plots: <root>/<pid>/analysis/lag/...")
    ap.add_argument("--pairs", default="wspd:DO2_c,salinity_psu:DO2_c,sst_c:bbp_532_m1,okubo_weiss:chl", help="Pairs 'x:y' comma-separated.")
    ap.add_argument("--max-lag-hours", type=int, default=None, help="Override YAML max_lag_hours")
    ap.add_argument("--lag-step-hours", type=int, default=None, help="Override YAML lag_step_hours")
    ap.add_argument("--tolerance-hours", type=float, default=None, help="Override YAML lag_match_tolerance_hours")
    ap.add_argument("--min-pairs", type=int, default=30, help="Min matched pairs to compute r at a lag")
    args = ap.parse_args()

    in_path = Path(args.input)
    cfg_path = Path(args.config)
    out_root = Path(args.outdir)
    plots_root = Path(args.plots_root)

    cfg = _load_yaml(cfg_path)
    analysis_cfg: Dict[str, Any] = (cfg or {}).get("analysis", {}) or {}
    coupling = (analysis_cfg.get("coupling", {}) or {})

    max_lag = int(args.max_lag_hours) if args.max_lag_hours is not None else int(coupling.get("max_lag_hours", 72))
    step = int(args.lag_step_hours) if args.lag_step_hours is not None else int(coupling.get("lag_step_hours", 3))
    tol = float(args.tolerance_hours) if args.tolerance_hours is not None else float(coupling.get("lag_match_tolerance_hours", 1.5))

    df = pd.read_csv(in_path)
    pid = _platform_id_from_df_or_filename(df, in_path)

    out_csv_dir = out_root / pid / "F_lag" / "lag"
    out_fig_dir = plots_root / pid / "analysis" / "lag"

    _ensure_dir(out_csv_dir)
    _ensure_dir(out_fig_dir)

    pairs = _parse_pairs_arg(args.pairs)

    log_lines = [
        f"INPUT: {in_path}",
        f"CONFIG: {cfg_path}",
        f"PLATFORM_ID: {pid}",
        "",
        f"max_lag_hours: {max_lag}",
        f"lag_step_hours: {step}",
        f"lag_match_tolerance_hours: {tol}",
        f"min_pairs: {args.min_pairs}",
        "",
        "pairs:",
        *[f"  - {x} : {y}" for x, y in pairs],
        "",
    ]

    for x, y in pairs:
        if x not in df.columns or y not in df.columns:
            log_lines.append(f"SKIP {x}:{y} (missing column)")
            continue

        res = lag_correlation_asof(
            df,
            x=x,
            y=y,
            max_lag_hours=max_lag,
            lag_step_hours=step,
            match_tolerance_hours=tol,
            min_pairs=int(args.min_pairs),
        )

        out_csv = out_csv_dir / f"lag_{res.pair_name}.csv"
        res.df.to_csv(out_csv, index=False)

        out_png = out_fig_dir / f"lag_{res.pair_name}.png"
        _plot_lag_curve(
            res.df,
            out_png,
            title=f"{pid} | lag corr {res.pair_name} | step={step}h tol={tol}h",
        )

        log_lines.append(f"OK {x}:{y} -> {out_csv.name}, {out_png.name}")

    _write_text(out_csv_dir / "run_log.txt", log_lines)

    print(f"[OK] platform_id={pid}")
    print(f"[OK] wrote lag outputs in: {out_csv_dir} and {out_fig_dir}")


if __name__ == "__main__":
    main()


"""
lag > 0: y viene dopo x → x anticipa y (possibile causalità “x → y”)
lag < 0: y anticipa x (più strano fisicamente in molti casi)

python -m bgcd.analysis.cli_lag_corr `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065378180/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots"

python -m bgcd.analysis.cli_lag_corr `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065379230/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots"

python -m bgcd.analysis.cli_lag_corr `
  --input "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT/300534065470010/B_window/data/master_subset_best_window.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots"
"""