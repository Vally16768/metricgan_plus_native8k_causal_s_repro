"""
Composite speech enhancement metrics from Hu & Loizou.

The implementation follows the coefficients from:
- Y. Hu and P. C. Loizou, "Evaluation of objective quality measures for
  speech enhancement," IEEE TASL, 2008.
- Philipos Loizou's reference MATLAB implementation (`composite.m`,
  `comp_wss.m`, `comp_llr.m`, `comp_snr.m`).
"""

from __future__ import annotations

import math

import numpy as np


_EPS = np.finfo(np.float64).eps
_ALPHA = 0.95
_NUM_CRIT = 25
_KMAX = 20.0
_KLOCMAX = 1.0
_MIN_SNR = -10.0
_MAX_SNR = 35.0
_CENT_FREQ = np.array(
    [
        50.0000,
        120.000,
        190.000,
        260.000,
        330.000,
        400.000,
        470.000,
        540.000,
        617.372,
        703.378,
        798.717,
        904.128,
        1020.38,
        1148.30,
        1288.72,
        1442.54,
        1610.70,
        1794.16,
        1993.93,
        2211.08,
        2446.71,
        2701.97,
        2978.04,
        3276.17,
        3597.63,
    ],
    dtype=np.float64,
)
_BANDWIDTH = np.array(
    [
        70.0000,
        70.0000,
        70.0000,
        70.0000,
        70.0000,
        70.0000,
        70.0000,
        77.3724,
        86.0056,
        95.3398,
        105.411,
        116.256,
        127.914,
        140.423,
        153.823,
        168.154,
        183.457,
        199.776,
        217.153,
        235.631,
        255.255,
        276.072,
        298.126,
        321.465,
        346.136,
    ],
    dtype=np.float64,
)


def _matlab_round(value: float) -> int:
    return int(math.floor(value + 0.5))


def _trim_mos(value: float) -> float:
    return float(np.clip(value, 1.0, 5.0))


def _trimmed_mean(values: np.ndarray, alpha: float = _ALPHA) -> float:
    if values.size == 0:
        raise ValueError("Metric requires at least one frame.")
    keep = max(1, _matlab_round(values.size * alpha))
    return float(np.mean(np.sort(values)[:keep]))


def composite_scores_from_components(
    *,
    pesq_mos: float,
    llr_mean: float,
    wss_dist: float,
    segsnr_mean: float,
) -> dict[str, float]:
    csig = 3.093 - (1.029 * llr_mean) + (0.603 * pesq_mos) - (0.009 * wss_dist)
    cbak = 1.634 + (0.478 * pesq_mos) - (0.007 * wss_dist) + (0.063 * segsnr_mean)
    covl = 1.594 + (0.805 * pesq_mos) - (0.512 * llr_mean) - (0.007 * wss_dist)
    return {
        "csig": _trim_mos(csig),
        "cbak": _trim_mos(cbak),
        "covl": _trim_mos(covl),
    }


def composite_scores(
    ref: np.ndarray,
    deg: np.ndarray,
    sr: int,
    *,
    pesq_value: float,
) -> dict[str, float]:
    if ref.ndim != 1 or deg.ndim != 1:
        raise ValueError("composite_scores expects 1D mono arrays.")
    if ref.shape != deg.shape:
        raise ValueError("ref and deg must have the same length.")

    ref64 = ref.astype(np.float64) + _EPS
    deg64 = deg.astype(np.float64) + _EPS
    wss_dist = _trimmed_mean(_wss_per_frame(ref64, deg64, sr))
    llr_mean = _trimmed_mean(_llr_per_frame(ref64, deg64, sr))
    segsnr_mean = float(np.mean(_segsnr_per_frame(ref64, deg64, sr)))
    return composite_scores_from_components(
        pesq_mos=float(pesq_value),
        llr_mean=llr_mean,
        wss_dist=wss_dist,
        segsnr_mean=segsnr_mean,
    )


def _frame_settings(sample_rate: int) -> tuple[int, int]:
    winlength = _matlab_round(30.0 * sample_rate / 1000.0)
    skiprate = math.floor(winlength / 4.0)
    if winlength <= 0 or skiprate <= 0:
        raise ValueError(f"Invalid frame settings for sample rate {sample_rate}.")
    return winlength, skiprate


def _num_frames(signal_length: int, winlength: int, skiprate: int) -> int:
    frames = int(math.floor((signal_length / skiprate) - (winlength / skiprate)))
    if frames <= 0:
        raise ValueError("Signal too short for composite metric framing.")
    return frames


def _analysis_window(winlength: int) -> np.ndarray:
    idx = np.arange(1, winlength + 1, dtype=np.float64)
    return 0.5 * (1.0 - np.cos((2.0 * np.pi * idx) / (winlength + 1.0)))


def _lpcoeff(speech_frame: np.ndarray, model_order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    winlength = int(speech_frame.size)
    r = np.empty(model_order + 1, dtype=np.float64)
    for k in range(model_order + 1):
        r[k] = np.sum(speech_frame[: winlength - k] * speech_frame[k:winlength])

    a = np.ones(model_order, dtype=np.float64)
    e = np.empty(model_order + 1, dtype=np.float64)
    e[0] = max(r[0], _EPS)
    rcoeff = np.zeros(model_order, dtype=np.float64)

    for i in range(model_order):
        if i == 0:
            sum_term = 0.0
            a_past = np.empty(0, dtype=np.float64)
        else:
            a_past = a[:i].copy()
            sum_term = np.sum(a_past * r[i:0:-1])
        rcoeff[i] = (r[i + 1] - sum_term) / max(e[i], _EPS)
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past - (rcoeff[i] * a_past[::-1])
        e[i + 1] = max((1.0 - (rcoeff[i] * rcoeff[i])) * e[i], _EPS)

    lpc_params = np.concatenate(([1.0], -a))
    return r, rcoeff, lpc_params


def _llr_per_frame(clean_speech: np.ndarray, processed_speech: np.ndarray, sample_rate: int) -> np.ndarray:
    if clean_speech.shape != processed_speech.shape:
        raise ValueError("Both speech signals must have the same length.")

    winlength, skiprate = _frame_settings(sample_rate)
    num_frames = _num_frames(clean_speech.size, winlength, skiprate)
    window = _analysis_window(winlength)
    order = 10 if sample_rate < 10000 else 16
    distortion = np.empty(num_frames, dtype=np.float64)

    for frame_idx in range(num_frames):
        start = frame_idx * skiprate
        clean_frame = clean_speech[start : start + winlength] * window
        processed_frame = processed_speech[start : start + winlength] * window
        r_clean, _, a_clean = _lpcoeff(clean_frame, order)
        _, _, a_processed = _lpcoeff(processed_frame, order)
        toeplitz = np.empty((order + 1, order + 1), dtype=np.float64)
        for row in range(order + 1):
            for col in range(order + 1):
                toeplitz[row, col] = r_clean[abs(row - col)]
        numerator = float(a_processed @ toeplitz @ a_processed.T)
        denominator = float(a_clean @ toeplitz @ a_clean.T)
        distortion[frame_idx] = math.log(max(numerator, _EPS) / max(denominator, _EPS))

    return distortion


def _critical_band_filters(sample_rate: int, winlength: int) -> tuple[np.ndarray, int]:
    max_freq = sample_rate / 2.0
    n_fft = 1 << math.ceil(math.log2(max(2, 2 * winlength)))
    n_fftby2 = n_fft // 2
    j = np.arange(n_fftby2, dtype=np.float64)
    bw_min = _BANDWIDTH[0]
    min_factor = math.exp(-30.0 / (2.0 * 2.303))
    crit_filter = np.empty((_NUM_CRIT, n_fftby2), dtype=np.float64)

    for band_idx in range(_NUM_CRIT):
        f0 = (_CENT_FREQ[band_idx] / max_freq) * n_fftby2
        bw = (_BANDWIDTH[band_idx] / max_freq) * n_fftby2
        norm_factor = math.log(bw_min) - math.log(_BANDWIDTH[band_idx])
        filt = np.exp((-11.0 * (((j - math.floor(f0)) / bw) ** 2.0)) + norm_factor)
        crit_filter[band_idx] = filt * (filt > min_factor)

    return crit_filter, n_fft


def _locate_peaks(energy: np.ndarray, slope: np.ndarray) -> np.ndarray:
    peaks = np.empty(_NUM_CRIT - 1, dtype=np.float64)
    for band_idx in range(_NUM_CRIT - 1):
        if slope[band_idx] > 0:
            n = band_idx
            while n < (_NUM_CRIT - 1) and slope[n] > 0:
                n += 1
            peaks[band_idx] = energy[max(n - 1, 0)]
        else:
            n = band_idx
            while n >= 0 and slope[n] <= 0:
                n -= 1
            peaks[band_idx] = energy[n + 1]
    return peaks


def _wss_per_frame(clean_speech: np.ndarray, processed_speech: np.ndarray, sample_rate: int) -> np.ndarray:
    if clean_speech.shape != processed_speech.shape:
        raise ValueError("Files must have the same length.")

    winlength, skiprate = _frame_settings(sample_rate)
    num_frames = _num_frames(clean_speech.size, winlength, skiprate)
    window = _analysis_window(winlength)
    crit_filter, n_fft = _critical_band_filters(sample_rate, winlength)
    n_fftby2 = n_fft // 2
    distortion = np.empty(num_frames, dtype=np.float64)

    for frame_idx in range(num_frames):
        start = frame_idx * skiprate
        clean_frame = clean_speech[start : start + winlength] * window
        processed_frame = processed_speech[start : start + winlength] * window

        clean_spec = np.abs(np.fft.fft(clean_frame, n_fft)) ** 2
        processed_spec = np.abs(np.fft.fft(processed_frame, n_fft)) ** 2

        clean_energy = 10.0 * np.log10(np.maximum(crit_filter @ clean_spec[:n_fftby2], 1e-10))
        processed_energy = 10.0 * np.log10(np.maximum(crit_filter @ processed_spec[:n_fftby2], 1e-10))

        clean_slope = clean_energy[1:] - clean_energy[:-1]
        processed_slope = processed_energy[1:] - processed_energy[:-1]
        clean_loc_peak = _locate_peaks(clean_energy, clean_slope)
        processed_loc_peak = _locate_peaks(processed_energy, processed_slope)

        dbmax_clean = float(np.max(clean_energy))
        dbmax_processed = float(np.max(processed_energy))

        wmax_clean = _KMAX / (_KMAX + dbmax_clean - clean_energy[:-1])
        wlocmax_clean = _KLOCMAX / (_KLOCMAX + clean_loc_peak - clean_energy[:-1])
        w_clean = wmax_clean * wlocmax_clean

        wmax_processed = _KMAX / (_KMAX + dbmax_processed - processed_energy[:-1])
        wlocmax_processed = _KLOCMAX / (_KLOCMAX + processed_loc_peak - processed_energy[:-1])
        w_processed = wmax_processed * wlocmax_processed

        weights = (w_clean + w_processed) / 2.0
        distortion[frame_idx] = np.sum(weights * ((clean_slope - processed_slope) ** 2))
        distortion[frame_idx] = distortion[frame_idx] / max(np.sum(weights), _EPS)

    return distortion


def _segsnr_per_frame(clean_speech: np.ndarray, processed_speech: np.ndarray, sample_rate: int) -> np.ndarray:
    if clean_speech.shape != processed_speech.shape:
        raise ValueError("Both speech signals must have the same length.")

    winlength, skiprate = _frame_settings(sample_rate)
    num_frames = _num_frames(clean_speech.size, winlength, skiprate)
    window = _analysis_window(winlength)
    segmental_snr = np.empty(num_frames, dtype=np.float64)

    for frame_idx in range(num_frames):
        start = frame_idx * skiprate
        clean_frame = clean_speech[start : start + winlength] * window
        processed_frame = processed_speech[start : start + winlength] * window
        signal_energy = float(np.sum(clean_frame**2))
        noise_energy = float(np.sum((clean_frame - processed_frame) ** 2))
        snr = 10.0 * math.log10((signal_energy / (noise_energy + _EPS)) + _EPS)
        segmental_snr[frame_idx] = float(np.clip(snr, _MIN_SNR, _MAX_SNR))

    return segmental_snr
