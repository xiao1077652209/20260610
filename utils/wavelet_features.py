"""Wavelet views for the dual-view spectral model."""
import numpy as np
import pywt


def _reconstruct_band(coeffs, wavelet, index, original_length):
    band_coeffs = [np.zeros_like(coeff) for coeff in coeffs]
    band_coeffs[index] = coeffs[index]
    reconstructed = pywt.waverec(band_coeffs, wavelet, mode="symmetric")
    return reconstructed[:original_length]


def build_wavelet_view(spectrum, wavelet="db4", level=3, include_denoised=False):
    """Return equal-length approximation and mid-frequency detail channels."""
    spectrum = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(spectrum), wavelet_obj.dec_len)
    actual_level = min(int(level), max_level)
    if actual_level < 1:
        raise ValueError(
            f"Spectrum length {len(spectrum)} is too short for wavelet {wavelet!r}."
        )

    coeffs = pywt.wavedec(spectrum, wavelet_obj, mode="symmetric", level=actual_level)
    approximation = _reconstruct_band(coeffs, wavelet_obj, 0, len(spectrum))

    # wavedec returns [cA_n, cD_n, ..., cD_1]. Retain middle/coarse details
    # and omit cD_1, which is usually dominated by high-frequency noise.
    detail_indices = range(1, len(coeffs) - 1) if len(coeffs) > 2 else range(1, len(coeffs))
    mid_detail = np.zeros_like(spectrum, dtype=np.float64)
    for index in detail_indices:
        mid_detail += _reconstruct_band(coeffs, wavelet_obj, index, len(spectrum))

    channels = [approximation, mid_detail]
    if include_denoised:
        channels.append(approximation + mid_detail)
    return np.stack(channels, axis=0).astype(np.float32)


def build_wavelet_views(spectra, wavelet="db4", level=3, include_denoised=False):
    spectra = np.asarray(spectra, dtype=np.float32)
    return np.stack([
        build_wavelet_view(row, wavelet, level, include_denoised)
        for row in spectra
    ], axis=0)
