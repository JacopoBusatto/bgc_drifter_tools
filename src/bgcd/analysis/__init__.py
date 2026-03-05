# ================================
# File: src/bgcd/analysis/__init__.py
# ================================
"""
bgcd.analysis

Analysis + QC tools for BGC-SVP MASTER datasets.
Lightweight exports only (no legacy preprocess imports).
"""
from .qc_core import (  # noqa: F401
    qc_time,
    qc_duplicates,
    qc_bounds,
    qc_coverage,
    QCTimeReport,
    QCDuplicatesReport,
    QCBoundsReport,
    QCCoverageReport,
)

from .overlap import (  # noqa: F401
    find_overlap_windows,
    OverlapResult,
    OverlapWindow,
)