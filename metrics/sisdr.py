# sisdr.py — STRICT
from __future__ import annotations
import numpy as np

def sisdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    if ref.ndim != 1 or est.ndim != 1:
        raise ValueError("sisdr expects 1D mono arrays.")
    if len(ref) != len(est):
        raise ValueError("ref and est must have the same length.")
    ref = ref.astype(np.float32)
    est = est.astype(np.float32)
    alpha = (np.dot(est, ref) + eps) / (np.dot(ref, ref) + eps)
    s_target = alpha * ref
    e_noise = est - s_target
    num = np.sum(s_target**2) + eps
    den = np.sum(e_noise**2) + eps
    return 10.0 * np.log10(num / den)
