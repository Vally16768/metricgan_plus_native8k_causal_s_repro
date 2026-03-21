# stoi.py — STRICT
from __future__ import annotations
import numpy as np
try:
    from pystoi import stoi as _stoi
except Exception as e:
    raise ImportError("The 'pystoi' package is required.") from e

def stoi_score(ref: np.ndarray, deg: np.ndarray, sr: int, extended: bool = False) -> float:
    if ref.ndim != 1 or deg.ndim != 1:
        raise ValueError("stoi_score expects 1D mono arrays.")
    val = float(_stoi(ref.astype(np.float32), deg.astype(np.float32), sr, extended=extended))
    if not (val == val):
        raise ValueError("STOI returned NaN.")
    return val
