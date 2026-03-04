# ================================
# File: src/bgcd/analysis/__init__.py
# ================================
"""
bgcd.analysis

Statistical analysis modules for BGC-SVP MASTER datasets.
Currently includes:
- preprocess: quality control, masking, time-axis checks
"""

from .preprocess import apply_qc, QCResult  # noqa: F401
