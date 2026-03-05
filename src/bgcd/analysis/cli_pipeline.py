# ================================
# File: src/bgcd/analysis/cli_pipeline.py
# ================================
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print("\n" + " ".join(cmd))
    p = subprocess.run(cmd, check=False)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bgcd.analysis.cli_pipeline",
        description="Run the full BGC-SVP analysis pipeline for one platform (QC/window -> coupling -> rolling -> lag -> PCA).",
    )

    # Inputs
    ap.add_argument("--master", required=True, help="Path to master_<platform>.csv")
    ap.add_argument("--config", required=True, help="analysis_config_min.yml")
    ap.add_argument("--outdir", required=True, help="Root OUT directory (like cli_window)")
    ap.add_argument("--plots-root", required=True, help="Root plots directory")
    ap.add_argument("--run-qc", default="true", help="true/false (passed to cli_window)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--export-topk", default="false", help="true/false")

    # Rolling/Lag params (overrides)
    ap.add_argument("--rolling-window-hours", type=int, default=48)
    ap.add_argument("--rolling-min-points", type=int, default=10)
    ap.add_argument("--with-trajectory-map", action="store_true")
    ap.add_argument("--decimate-quiver", type=int, default=7)

    # PCA cluster params
    ap.add_argument("--cluster-k", type=int, default=3)
    ap.add_argument("--cluster-use-pcs", type=int, nargs="+", default=[1, 2])

    # Controls
    ap.add_argument("--skip-window", action="store_true")
    ap.add_argument("--skip-coupling", action="store_true")
    ap.add_argument("--skip-rolling", action="store_true")
    ap.add_argument("--skip-lag", action="store_true")
    ap.add_argument("--skip-pca", action="store_true")
    ap.add_argument("--skip-pca-clustering", action="store_true")

    args = ap.parse_args()

    master = Path(args.master)
    config = Path(args.config)
    outdir = Path(args.outdir)
    plots_root = Path(args.plots_root)

    # 1) window selection (writes OUT/<pid>/B_window/data/master_subset_best_window.csv)
    if not args.skip_window:
        _run(
            [
                sys.executable,
                "-m",
                "bgcd.analysis.cli_window",
                "--master",
                str(master),
                "--config",
                str(config),
                "--outdir",
                str(outdir),
                "--run-qc",
                str(args.run_qc).lower(),
                "--topk",
                str(args.topk),
                "--export-topk",
                str(args.export_topk).lower(),
            ]
        )

    # Infer platform_id from filename: master_<pid>.csv
    # (robust enough given your naming; if not, we can parse the produced OUT folder)
    stem = master.stem
    pid = stem.replace("master_", "")
    subset = outdir / pid / "B_window" / "data" / "master_subset_best_window.csv"
    if not subset.exists():
        raise SystemExit(f"[ERR] Expected subset not found: {subset}")

    # 2) coupling
    if not args.skip_coupling:
        _run(
            [
                sys.executable,
                "-m",
                "bgcd.analysis.cli_coupling",
                "--input",
                str(subset),
                "--config",
                str(config),
                "--outdir",
                str(outdir),
                "--plots-root",
                str(plots_root),
            ]
        )

    # 3) rolling corr
    if not args.skip_rolling:
        cmd = [
            sys.executable,
            "-m",
            "bgcd.analysis.cli_rolling_corr",
            "--input",
            str(subset),
            "--config",
            str(config),
            "--outdir",
            str(outdir),
            "--plots-root",
            str(plots_root),
            "--window-hours",
            str(args.rolling_window_hours),
            "--min-points",
            str(args.rolling_min_points),
            "--decimate-quiver",
            str(args.decimate_quiver),
        ]
        if args.with_trajectory_map:
            cmd.append("--with-trajectory-map")
        _run(cmd)

    # 4) lag corr
    if not args.skip_lag:
        _run(
            [
                sys.executable,
                "-m",
                "bgcd.analysis.cli_lag_corr",
                "--input",
                str(subset),
                "--config",
                str(config),
                "--outdir",
                str(outdir),
                "--plots-root",
                str(plots_root),
            ]
        )

    # 5) PCA
    if not args.skip_pca:
        _run(
            [
                sys.executable,
                "-m",
                "bgcd.analysis.cli_pca",
                "--input",
                str(subset),
                "--config",
                str(config),
                "--outdir",
                str(outdir),
                "--plots-root",
                str(plots_root),
            ]
        )

    if not args.skip_pca_clustering:
        print(subset)
        _run(
            [
                sys.executable,
                "-m",
                "bgcd.analysis.cli_pca_cluster",
                "--input",
                str(subset),
                "--config",
                str(config),
                "--out-root",
                str(outdir),
                "--plots-root",
                str(plots_root),
                "--k",
                str(args.cluster_k),
                "--use-pcs",
                *[str(x) for x in args.cluster_use_pcs],
            ]
        )



if __name__ == "__main__":
    main()

"""

python -m bgcd.analysis.cli_pipeline `
  --master "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065378180.csv" `
  --config "analysis_config/analysis_config_min_no_sal.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --with-trajectory-map `
  --rolling-window-hours 48 `
  --rolling-min-points 10 `
  --decimate-quiver 7 `
  --cluster-k 3 `
  --cluster-use-pcs 1 2 3

python -m bgcd.analysis.cli_pipeline `
  --master "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065379230.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --with-trajectory-map `
  --rolling-window-hours 48 `
  --rolling-min-points 10 `
  --decimate-quiver 7 `
  --cluster-k 3 `
  --cluster-use-pcs 1 2 3
  

python -m bgcd.analysis.cli_pipeline `
  --master "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065470010.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --plots-root "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/plots" `
  --with-trajectory-map `
  --rolling-window-hours 48 `
  --rolling-min-points 10 `
  --decimate-quiver 7 `
  --cluster-k 3 `
  --cluster-use-pcs 1 2 3
"""