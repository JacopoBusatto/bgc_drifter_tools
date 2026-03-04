# ================================
# File: src/bgcd/cli_qc.py
# ================================
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from bgcd.analysis.preprocess import apply_qc


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load YAML config.

    Requires PyYAML (pip install pyyaml).
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: PyYAML. Install with: pip install pyyaml"
        ) from e

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping/dict, got {type(cfg)}")

    return cfg


def _print_qc_console_summary(qc_summary: pd.DataFrame, warnings: list[str], *, max_warnings: int = 15) -> None:
    print("\n=== QC SUMMARY (table) ===")
    if qc_summary.empty:
        print("(empty)")
    else:
        # Print nicely without index
        print(qc_summary.to_string(index=False))

    print("\n=== QC WARNINGS (first lines) ===")
    if not warnings:
        print("(none)")
        return

    for w in warnings[:max_warnings]:
        print(w)

    if len(warnings) > max_warnings:
        print(f"... ({len(warnings) - max_warnings} more warnings)")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="bgcd.cli_qc",
        description="Run Quality Control (QC) on a MASTER CSV and write a QC report.",
    )
    p.add_argument(
        "--master",
        required=True,
        help="Path to MASTER CSV (single platform_id per file, recommended).",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to analysis YAML config (e.g., analysis_config/analysis_config_min.yml).",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory where QC report folder will be created.",
    )
    p.add_argument(
        "--time-col",
        default="time_utc",
        help="Time column name in the MASTER CSV (default: time_utc).",
    )
    p.add_argument(
        "--group-col",
        default="platform_id",
        help="Platform column name in the MASTER CSV (default: platform_id).",
    )
    p.add_argument(
        "--write-clean",
        default=None,
        help=(
            "Optional: write cleaned MASTER CSV to this path. "
            "If a directory is provided, file will be saved as <outdir>/<platform_id>_master_clean.csv."
        ),
    )
    p.add_argument(
        "--csv-sep",
        default=",",
        help="CSV separator (default: ',').",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding (default: utf-8). Try 'latin1' if needed.",
    )

    args = p.parse_args()

    master_path = Path(args.master)
    cfg_path = Path(args.config)
    outdir = Path(args.outdir)

    if not master_path.exists():
        raise FileNotFoundError(f"MASTER not found: {master_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = _load_yaml(cfg_path)

    # Read MASTER CSV
    # Note: do NOT parse_dates here; apply_qc will coerce time_col to datetime.
    df = pd.read_csv(master_path, sep=args.csv_sep, encoding=args.encoding)

    # Run QC
    df_clean, qc = apply_qc(
        df,
        cfg,
        time_col=args.time_col,
        group_col=args.group_col,
        outdir=outdir,
    )

    # Console summary
    _print_qc_console_summary(qc.summary, qc.warnings, max_warnings=15)

    # Optionally write cleaned CSV
    if args.write_clean is not None:
        write_target = Path(args.write_clean)

        # If user passed a directory, construct file name
        if write_target.exists() and write_target.is_dir():
            # Use the folder where qc was written if available
            pid = qc.outdir.name if qc.outdir is not None else "__UNKNOWN__"
            out_csv = write_target / f"{pid}_master_clean.csv"
        elif str(write_target).endswith(("/", "\\")) or write_target.suffix == "":
            # Treat as directory-like path even if it doesn't exist yet
            write_target.mkdir(parents=True, exist_ok=True)
            pid = qc.outdir.name if qc.outdir is not None else "__UNKNOWN__"
            out_csv = write_target / f"{pid}_master_clean.csv"
        else:
            # Treat as file path
            write_target.parent.mkdir(parents=True, exist_ok=True)
            out_csv = write_target

        df_clean.to_csv(out_csv, index=False)
        print(f"\n[QC] Cleaned MASTER written to: {out_csv}")

    # Print where QC report is
    if qc.outdir is not None:
        print(f"\n[QC] QC report written under: {qc.outdir}")
    else:
        print("\n[QC] QC report writing disabled or outdir not provided.")


if __name__ == "__main__":
    main()


"""
python -m bgcd.cli_qc --master "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065379230.csv" --config "analysis_config/analysis_config_min.yml" --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/OUT_ANALYSIS"
"""