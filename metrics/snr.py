from __future__ import annotations
import numpy as np

_EPS = 1e-12

def _snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    """
    SNR(clean, test) in dB.
    """
    c = clean.astype(np.float64).ravel()
    t = test.astype(np.float64).ravel()
    n = min(len(c), len(t))
    if n == 0:
        raise ValueError("Empty input")
    c = c[:n]
    t = t[:n]
    nrg = float(np.sum(c * c) + _EPS)
    err = c - t
    den = float(np.sum(err * err) + _EPS)
    return 10.0 * np.log10(nrg / den)


def snr_noisy(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    SNR la intrare: SNR(clean, noisy)
    """
    return _snr_db(clean, noisy)


def snr_enhanced(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """
    SNR la ieșire: SNR(clean, enhanced)
    """
    return _snr_db(clean, enhanced)


def delta_snr(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Îmbunătățirea SNR (ΔSNR) = SNR_OUT - SNR_IN.

    SNR_IN  = SNR(clean, noisy)
    SNR_OUT = SNR(clean, enhanced)
    """
    snr_in = snr_noisy(clean, noisy)
    snr_out = snr_enhanced(clean, enhanced)
    return float(snr_out - snr_in)
