# ================================
# File: src/bgcd/analysis/pca.py
# ================================
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    variables: List[str]
    loadings: pd.DataFrame          # index: var, columns: PC1..PCk
    scores: pd.DataFrame            # columns: time_utc + PC1..PCk (+ optional lat/lon)
    explained: pd.DataFrame         # PC, explained_var_ratio, explained_var_ratio_cum
    n_rows_in: int
    n_rows_used: int


def run_pca(
    df: pd.DataFrame,
    *,
    variables: List[str],
    time_col: str = "time_utc",
    keep_cols: List[str] | None = None,   # e.g. ["lat","lon"]
    n_components: int | None = None,
) -> PCAResult:
    """
    Run PCA on selected variables:
      - drop rows with NaN among variables (no interpolation)
      - z-score standardization
      - PCA (default: all components)

    Returns:
      - loadings: pca.components_.T
      - scores: PC scores per row, aligned with time_utc (and keep_cols)
      - explained: explained variance ratio and cumulative
    """
    keep_cols = keep_cols or []
    n_rows_in = len(df)

    vars_present = [v for v in variables if v in df.columns]
    if not vars_present:
        raise ValueError("No PCA variables found in dataframe (all missing columns).")

    cols_needed = [time_col] + keep_cols + vars_present
    d = df[cols_needed].copy()

    d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)
    d = d.dropna(subset=[time_col])

    # drop rows with missing in PCA vars
    d_pca = d.dropna(subset=vars_present).copy()
    n_rows_used = len(d_pca)
    if n_rows_used < 5:
        raise ValueError(f"Too few rows for PCA after dropna: {n_rows_used}")

    X = d_pca[vars_present].to_numpy(dtype=float)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    scores_arr = pca.fit_transform(Xz)

    k = scores_arr.shape[1]
    pc_cols = [f"PC{i+1}" for i in range(k)]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=vars_present,
        columns=pc_cols,
    )

    scores = pd.DataFrame(scores_arr, columns=pc_cols)
    scores.insert(0, time_col, d_pca[time_col].to_numpy())
    for c in keep_cols:
        if c in d_pca.columns:
            scores[c] = d_pca[c].to_numpy()

    evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
    explained = pd.DataFrame(
        {
            "PC": pc_cols,
            "explained_var_ratio": evr,
            "explained_var_ratio_cum": np.cumsum(evr),
        }
    )

    return PCAResult(
        variables=vars_present,
        loadings=loadings,
        scores=scores,
        explained=explained,
        n_rows_in=n_rows_in,
        n_rows_used=n_rows_used,
    )