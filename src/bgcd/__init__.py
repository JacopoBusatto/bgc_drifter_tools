"""
bgcd: Tools for preprocessing and analyzing biogeochemical drifter datasets.
"""

from __future__ import annotations
from .io import read_drifter_csv, read_wind_csv, merge_drifter_wind
from .mat_io import inspect_mat, load_mat_any

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("bgcd")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__", "read_drifter_csv", "read_wind_csv", "merge_drifter_wind"]
