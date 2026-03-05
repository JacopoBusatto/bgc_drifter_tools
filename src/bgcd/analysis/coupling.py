# ================================
# File: src/bgcd/analysis/coupling.py
# ================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PairwiseCorrResult:
    table: pd.DataFrame          # one row per (x,y)
    used_phys: List[str]
    used_bio: List[str]
    warnings: List[str]


def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    # assumes finite and same length >= 2
    if x.size < 2:
        return np.nan
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.sqrt((x0 * x0).sum()) * np.sqrt((y0 * y0).sum())
    if den == 0:
        return np.nan
    return float((x0 * y0).sum() / den)


def pairwise_pearson_phys_bio(
    df: pd.DataFrame,
    *,
    phys_vars: List[str],
    bio_vars: List[str],
    min_fraction: float = 0.7,
    min_points: int = 50,
) -> PairwiseCorrResult:
    """
    Compute Pearson correlation for each phys×bio pair using overlap (both non-NaN).
    - If overlap fraction < min_fraction OR overlap points < min_points: pair is skipped (r=NaN, status='skip').
    - Overlap fraction is computed relative to the available rows for the run (len(df)).
    """
    phys_vars = _dedup_keep_order([v for v in phys_vars if v in df.columns])
    bio_vars = _dedup_keep_order([v for v in bio_vars if v in df.columns])

    warnings: List[str] = []
    n_total = len(df)

    if n_total == 0:
        return PairwiseCorrResult(
            table=pd.DataFrame(columns=["x", "y", "n_overlap", "frac_overlap", "pearson_r", "status", "reason"]),
            used_phys=phys_vars,
            used_bio=bio_vars,
            warnings=["Empty dataframe: nothing to analyze."],
        )

    if not phys_vars:
        warnings.append("No phys_vars present in dataframe (all missing).")
    if not bio_vars:
        warnings.append("No bio_vars present in dataframe (all missing).")

    rows = []
    for xname in phys_vars:
        x = df[xname]
        for yname in bio_vars:
            y = df[yname]

            ok = x.notna() & y.notna()
            n = int(ok.sum())
            frac = float(n / n_total) if n_total > 0 else 0.0

            if n < min_points:
                rows.append(
                    {
                        "x": xname,
                        "y": yname,
                        "n_overlap": n,
                        "frac_overlap": frac,
                        "pearson_r": np.nan,
                        "status": "skip",
                        "reason": f"n_overlap<{min_points}",
                    }
                )
                continue

            if frac < min_fraction:
                rows.append(
                    {
                        "x": xname,
                        "y": yname,
                        "n_overlap": n,
                        "frac_overlap": frac,
                        "pearson_r": np.nan,
                        "status": "skip",
                        "reason": f"frac_overlap<{min_fraction}",
                    }
                )
                continue

            xv = x[ok].to_numpy(dtype=float)
            yv = y[ok].to_numpy(dtype=float)
            r = _pearson_r(xv, yv)
            rows.append(
                {
                    "x": xname,
                    "y": yname,
                    "n_overlap": n,
                    "frac_overlap": frac,
                    "pearson_r": r,
                    "status": "ok",
                    "reason": "",
                }
            )

    table = pd.DataFrame(rows)
    if not table.empty:
        # sort: strongest abs(r) first among ok pairs
        ok_mask = table["status"] == "ok"
        table_ok = table.loc[ok_mask].copy()
        table_skip = table.loc[~ok_mask].copy()
        if not table_ok.empty:
            table_ok["abs_r"] = table_ok["pearson_r"].abs()
            table_ok = table_ok.sort_values(["abs_r", "n_overlap"], ascending=[False, False]).drop(columns=["abs_r"])
        table = pd.concat([table_ok, table_skip], ignore_index=True)

    # global warnings about missingness (useful hint to promote required)
    # Example: if a variable exists but has too few valid points overall, warn.
    for v in _dedup_keep_order(phys_vars + bio_vars):
        frac_valid = float(df[v].notna().mean()) if v in df.columns else 0.0
        if frac_valid < min_fraction:
            warnings.append(
                f"Variable '{v}' has low valid fraction ({frac_valid:.2f}) in this window. "
                "If it is essential, consider adding it to data_window.required."
            )

    return PairwiseCorrResult(table=table, used_phys=phys_vars, used_bio=bio_vars, warnings=warnings)