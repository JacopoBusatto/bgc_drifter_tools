from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _try_load_scipy_mat(path: str | Path) -> Dict[str, Any] | None:
    try:
        from scipy.io import loadmat
    except Exception:
        return None

    p = str(path)
    try:
        d = loadmat(p, squeeze_me=True, struct_as_record=False)
        # remove meta-keys
        d = {k: v for k, v in d.items() if not k.startswith("__")}
        return d
    except Exception:
        return None


def _try_load_h5_mat(path: str | Path) -> Dict[str, Any] | None:
    try:
        import h5py
    except Exception:
        return None

    p = str(path)
    try:
        out: Dict[str, Any] = {}
        with h5py.File(p, "r") as f:
            def _walk(name: str, obj: Any) -> None:
                # keep only datasets at leaf nodes
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    out[name] = obj[()]
            f.visititems(_walk)
        return out
    except Exception:
        return None


def load_mat_any(path: str | Path) -> Dict[str, Any]:
    """
    Load .mat files with either scipy (MAT v5/v7) or h5py (MAT v7.3).
    Returns a dict of variables.
    """
    d = _try_load_scipy_mat(path)
    if d is not None:
        return d

    d = _try_load_h5_mat(path)
    if d is not None:
        return d

    raise RuntimeError(
        "Unable to load .mat. Install scipy and h5py and ensure the file is a valid MAT v5/v7 or v7.3."
    )


def inspect_mat(path: str | Path, max_items: int = 50) -> None:
    """
    Print keys and basic info (dtype, shape, min/max for numeric arrays).
    """
    d = load_mat_any(path)
    keys = sorted(d.keys())

    print(f"File: {path}")
    print(f"Keys ({len(keys)}):")
    for i, k in enumerate(keys[:max_items]):
        v = d[k]
        if isinstance(v, np.ndarray):
            info = f"ndarray dtype={v.dtype} shape={v.shape}"
            if np.issubdtype(v.dtype, np.number) and v.size > 0:
                vv = v.astype(float).ravel()
                info += f" range=[{np.nanmin(vv):.3g}, {np.nanmax(vv):.3g}]"
            print(f"  - {k}: {info}")
        else:
            print(f"  - {k}: {type(v)}")
    if len(keys) > max_items:
        print(f"... ({len(keys) - max_items} more)")
