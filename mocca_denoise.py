import numpy as np
from scipy.signal import savgol_filter
import pywt


def savgol_denoise(y, window=11, poly=3):
    """
    Shape-preserving local polynomial smoothing.
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 5:
        return y.copy()

    if window >= len(y):
        # if too short, just return as-is
        return y.copy()

    if window % 2 == 0:
        window += 1

    return savgol_filter(y, window_length=window, polyorder=poly)


def wavelet_denoise(y, wavelet="db6", level=3):
    """
    Global denoising via wavelet shrinkage (adaptive level to avoid boundary artifacts).
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 32:
        # too short to wavelet denoise safely
        return y.copy()

    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=n, filter_len=w.dec_len)
    use_level = max(1, min(level, max_level))

    coeffs = pywt.wavedec(y, wavelet, level=use_level)

    # Robust noise estimate from highest-frequency detail
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 if len(detail) > 0 else 0.0
    if sigma <= 0:
        return y.copy()

    threshold = sigma * np.sqrt(2 * np.log(n))

    coeffs_denoised = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_denoised.append(pywt.threshold(c, threshold, mode="soft"))

    y_dn = pywt.waverec(coeffs_denoised, wavelet)
    return y_dn[:n]


def denoise_signal(y, preset="medium"):
    """
    Preset-based denoising interface.
    This is the ONLY function the pipeline should call.
    """
    if preset == "gentle":
        return savgol_denoise(y, window=7, poly=2)

    if preset == "medium":
        y1 = wavelet_denoise(y, wavelet="db6", level=3)
        return savgol_denoise(y1, window=11, poly=3)

    if preset == "strong":
        y1 = wavelet_denoise(y, wavelet="db8", level=4)
        return savgol_denoise(y1, window=15, poly=3)

    raise ValueError(f"Unknown denoise preset: {preset}")
