"""
Wavelet denoising for financial time series.

Implements the Haar à Trous Wavelet (HTW) non-decimated discrete wavelet
transform following Jammazi, Lahiani & Nguyen (2015) and Murtagh et al. (2004).

Also provides wrappers for PyWavelets (MODWT/DWT) for alternative decompositions.

The denoising procedure:
1. Decompose signal using HTW into smoothed (s_J) and detail (d_j) coefficients
2. Apply Donoho (1995) universal threshold: λ = √(2σ²·log(N))
3. Soft/hard threshold the detail coefficients
4. Reconstruct the denoised signal

References
----------
Jammazi, R., Lahiani, A., & Nguyen, D. K. (2015). A wavelet-based nonlinear
    ARDL model for assessing the exchange rate pass-through to crude oil prices.
    J. Int. Fin. Markets, Inst. and Money, 34, 173-187.

Murtagh, F., Starck, J. L., & Renaud, O. (2004). On neuro-wavelet modeling.
    Decision Support Systems, 37, 475-484.

Donoho, D. L. (1995). De-noising by soft-thresholding.
    IEEE Trans. Inf. Theory, 41, 613-627.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
import warnings

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


# ══════════════════════════════════════════════════════════════════════
# Haar à Trous Wavelet Transform (HTW)
# ══════════════════════════════════════════════════════════════════════

class HaarATrousWavelet:
    """
    Non-decimated Haar à Trous Wavelet Transform.

    This is a redundant (translation-invariant) DWT using the simple
    non-symmetric filter h = (1/2, 1/2) as described in Murtagh et al. (2004).

    At each level j:
        s_{j+1}(t) = 0.5 * [s_j(t - 2^j) + s_j(t)]     (Eq. 8 in paper)
        d_{j+1}(t) = s_j(t) - s_{j+1}(t)                 (Eq. 9 in paper)

    The signal can be reconstructed as:
        x(t) = s_J(t) + Σ_{j=1}^{J} d_j(t)               (Eq. 7 in paper)

    Parameters
    ----------
    n_levels : int or None
        Number of decomposition levels. If None, uses floor(log2(N)).
    threshold_method : str
        'soft' or 'hard' thresholding (default: 'soft').
    threshold_rule : str
        'donoho' — universal threshold λ = √(2σ²·log(N))
        'custom' — user-provided threshold value
    """

    def __init__(
        self,
        n_levels: Optional[int] = None,
        threshold_method: str = 'soft',
        threshold_rule: str = 'donoho',
    ):
        self.n_levels = n_levels
        self.threshold_method = threshold_method.lower()
        self.threshold_rule = threshold_rule.lower()

    def decompose(
        self, signal: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Decompose signal into smooth and detail coefficients.

        Parameters
        ----------
        signal : 1-D array

        Returns
        -------
        smooth : np.ndarray
            Smoothed signal at the coarsest level (s_J).
        details : list of np.ndarray
            Detail coefficients [d_1, d_2, ..., d_J].
        """
        x = np.asarray(signal, dtype=float).copy()
        n = len(x)

        if self.n_levels is None:
            J = int(np.floor(np.log2(n)))
        else:
            J = self.n_levels

        details = []
        s_prev = x.copy()  # s_0 = original signal

        for j in range(J):
            shift = 2 ** j
            s_curr = np.zeros(n)

            for t in range(n):
                t_shifted = max(0, t - shift)
                s_curr[t] = 0.5 * (s_prev[t_shifted] + s_prev[t])

            d_j = s_prev - s_curr  # detail at level j+1
            details.append(d_j)
            s_prev = s_curr.copy()

        smooth = s_prev
        return smooth, details

    def _estimate_noise_sigma(self, detail_level1: np.ndarray) -> float:
        """
        Estimate noise standard deviation from first-level detail coefficients.

        Uses MAD (Median Absolute Deviation) estimator:
            σ = MAD(d_1) / 0.6745
        """
        mad = np.median(np.abs(detail_level1 - np.median(detail_level1)))
        return mad / 0.6745

    def _compute_threshold(
        self, detail: np.ndarray, n: int, sigma: float
    ) -> float:
        """
        Compute threshold value.

        Donoho (1995): λ = √(2σ²·log(N))
        """
        return sigma * np.sqrt(2 * np.log(n))

    def _apply_threshold(
        self, coefficients: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Apply soft or hard thresholding to wavelet coefficients."""
        if self.threshold_method == 'soft':
            return np.sign(coefficients) * np.maximum(
                np.abs(coefficients) - threshold, 0
            )
        elif self.threshold_method == 'hard':
            result = coefficients.copy()
            result[np.abs(result) < threshold] = 0
            return result
        else:
            raise ValueError(
                f"Unknown threshold method: {self.threshold_method}")

    def denoise(
        self,
        signal: np.ndarray,
        custom_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Denoise a signal using HTW decomposition and thresholding.

        Parameters
        ----------
        signal : 1-D array
        custom_threshold : float, optional
            Custom threshold value (only with threshold_rule='custom').

        Returns
        -------
        denoised : np.ndarray
            The denoised (smoothed) signal.
        """
        x = np.asarray(signal, dtype=float)
        n = len(x)

        smooth, details = self.decompose(x)

        # Estimate noise from first detail level
        sigma = self._estimate_noise_sigma(details[0])

        if self.threshold_rule == 'donoho':
            threshold = self._compute_threshold(details[0], n, sigma)
        elif self.threshold_rule == 'custom' and custom_threshold is not None:
            threshold = custom_threshold
        else:
            threshold = self._compute_threshold(details[0], n, sigma)

        # Apply thresholding to all detail levels
        thresholded_details = []
        for d in details:
            thresholded_details.append(self._apply_threshold(d, threshold))

        # Reconstruct: x_denoised = s_J + Σ d_j_thresholded
        denoised = smooth.copy()
        for d in thresholded_details:
            denoised += d

        return denoised

    def full_analysis(
        self, signal: np.ndarray, name: str = 'signal'
    ) -> Dict:
        """
        Perform complete wavelet analysis: decompose, denoise, and return
        all components.

        Returns
        -------
        dict with keys:
            'original', 'denoised', 'smooth', 'details',
            'thresholded_details', 'threshold', 'sigma', 'n_levels'
        """
        x = np.asarray(signal, dtype=float)
        n = len(x)

        smooth, details = self.decompose(x)
        sigma = self._estimate_noise_sigma(details[0])
        threshold = self._compute_threshold(details[0], n, sigma)

        thresholded = [self._apply_threshold(d, threshold) for d in details]

        denoised = smooth.copy()
        for d in thresholded:
            denoised += d

        return {
            'name': name,
            'original': x,
            'denoised': denoised,
            'smooth': smooth,
            'details': details,
            'thresholded_details': thresholded,
            'threshold': threshold,
            'sigma': sigma,
            'n_levels': len(details),
        }


# ══════════════════════════════════════════════════════════════════════
# PyWavelets-based methods (MODWT / DWT wrappers)
# ══════════════════════════════════════════════════════════════════════

class PyWaveletDenoiser:
    """
    Wavelet denoising using PyWavelets library.

    Supports MODWT, DWT, and SWT for comparison with the HTW method.

    Parameters
    ----------
    wavelet : str
        Wavelet name (default: 'haar'). See pywt.wavelist() for options.
    mode : str
        Signal extension mode (default: 'symmetric').
    n_levels : int or None
        Decomposition levels.
    method : str
        'swt' (Stationary WT, non-decimated) or 'dwt' (Decimated WT).
    threshold_method : str
        'soft' or 'hard'.
    """

    def __init__(
        self,
        wavelet: str = 'haar',
        mode: str = 'symmetric',
        n_levels: Optional[int] = None,
        method: str = 'swt',
        threshold_method: str = 'soft',
    ):
        if not HAS_PYWT:
            raise ImportError(
                "PyWavelets is required. Install with: pip install PyWavelets")
        self.wavelet = wavelet
        self.mode = mode
        self.n_levels = n_levels
        self.method = method.lower()
        self.threshold_method = threshold_method.lower()

    def decompose(self, signal: np.ndarray) -> Dict:
        """Decompose signal using PyWavelets."""
        x = np.asarray(signal, dtype=float)

        if self.n_levels is None:
            max_level = pywt.swt_max_level(len(x))
            n_levels = min(max_level, int(np.floor(np.log2(len(x)))))
        else:
            n_levels = self.n_levels

        if self.method == 'swt':
            # Stationary Wavelet Transform (non-decimated)
            # Pad to power of 2 if needed
            orig_len = len(x)
            pad_len = int(2 ** np.ceil(np.log2(orig_len)))
            if pad_len > orig_len:
                x_padded = np.pad(x, (0, pad_len - orig_len), mode='reflect')
            else:
                x_padded = x

            coeffs = pywt.swt(x_padded, self.wavelet, level=n_levels)
            # coeffs is list of (cA, cD) tuples from coarsest to finest
            details = [c[1][:orig_len] for c in coeffs]
            approx = coeffs[0][0][:orig_len]  # coarsest approximation

            return {
                'approximation': approx,
                'details': details[::-1],  # finest to coarsest
                'n_levels': n_levels,
                'method': 'swt',
            }
        elif self.method == 'dwt':
            coeffs = pywt.wavedec(x, self.wavelet, mode=self.mode, level=n_levels)
            approx = coeffs[0]
            details = coeffs[1:]
            return {
                'approximation': approx,
                'details': details,
                'n_levels': n_levels,
                'method': 'dwt',
            }
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'swt' or 'dwt'.")

    def denoise(
        self, signal: np.ndarray,
        threshold_rule: str = 'donoho',
    ) -> np.ndarray:
        """
        Denoise signal using PyWavelets.

        Parameters
        ----------
        signal : 1-D array
        threshold_rule : str
            'donoho' — universal threshold

        Returns
        -------
        denoised : np.ndarray
        """
        x = np.asarray(signal, dtype=float)
        n = len(x)

        if self.method == 'swt':
            orig_len = n
            pad_len = int(2 ** np.ceil(np.log2(n)))
            if pad_len > n:
                x_padded = np.pad(x, (0, pad_len - n), mode='reflect')
            else:
                x_padded = x

            n_levels = self.n_levels or min(
                pywt.swt_max_level(len(x_padded)),
                int(np.floor(np.log2(n)))
            )

            coeffs = pywt.swt(x_padded, self.wavelet, level=n_levels)

            # Estimate sigma from finest detail
            finest_detail = coeffs[-1][1]
            sigma = np.median(np.abs(finest_detail)) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(x_padded)))

            # Threshold detail coefficients
            new_coeffs = []
            for ca, cd in coeffs:
                if self.threshold_method == 'soft':
                    cd_new = pywt.threshold(cd, threshold, mode='soft')
                else:
                    cd_new = pywt.threshold(cd, threshold, mode='hard')
                new_coeffs.append((ca, cd_new))

            denoised = pywt.iswt(new_coeffs, self.wavelet)
            return denoised[:orig_len]

        elif self.method == 'dwt':
            n_levels = self.n_levels or pywt.dwt_max_level(n, self.wavelet)
            coeffs = pywt.wavedec(x, self.wavelet, mode=self.mode, level=n_levels)

            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(n))

            new_coeffs = [coeffs[0]]  # keep approximation
            for cd in coeffs[1:]:
                if self.threshold_method == 'soft':
                    new_coeffs.append(pywt.threshold(cd, threshold, mode='soft'))
                else:
                    new_coeffs.append(pywt.threshold(cd, threshold, mode='hard'))

            return pywt.waverec(new_coeffs, self.wavelet, mode=self.mode)[:n]
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def full_analysis(self, signal: np.ndarray, name: str = 'signal') -> Dict:
        """Complete wavelet analysis with all components."""
        x = np.asarray(signal, dtype=float)
        dec = self.decompose(x)
        denoised = self.denoise(x)

        return {
            'name': name,
            'original': x,
            'denoised': denoised,
            'decomposition': dec,
            'wavelet': self.wavelet,
            'method': self.method,
            'n_levels': dec['n_levels'],
        }


# ══════════════════════════════════════════════════════════════════════
# Convenience function
# ══════════════════════════════════════════════════════════════════════

def denoise_series(
    series: np.ndarray,
    method: str = 'htw',
    wavelet: str = 'haar',
    n_levels: Optional[int] = None,
    threshold_method: str = 'soft',
) -> np.ndarray:
    """
    One-line convenience function to denoise a time series.

    Parameters
    ----------
    series : array-like
    method : str
        'htw' — Haar à Trous (paper method, default)
        'swt' — Stationary Wavelet Transform (PyWavelets)
        'dwt' — Discrete Wavelet Transform (PyWavelets)
    wavelet : str
        Wavelet name for PyWavelets methods.
    n_levels : int or None
    threshold_method : str
        'soft' or 'hard'

    Returns
    -------
    denoised : np.ndarray
    """
    x = np.asarray(series, dtype=float)

    if method.lower() == 'htw':
        wt = HaarATrousWavelet(
            n_levels=n_levels,
            threshold_method=threshold_method,
        )
        return wt.denoise(x)
    elif method.lower() in ('swt', 'dwt'):
        wt = PyWaveletDenoiser(
            wavelet=wavelet,
            n_levels=n_levels,
            method=method.lower(),
            threshold_method=threshold_method,
        )
        return wt.denoise(x)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'htw', 'swt', or 'dwt'.")


def denoise_dataframe(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'htw',
    prefix: str = 'S',
    **kwargs,
) -> pd.DataFrame:
    """
    Denoise multiple columns of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str or None (all numeric columns)
    method : str
    prefix : str
        Prefix for denoised column names (default: 'S' for Smoothed).

    Returns
    -------
    pd.DataFrame with original and denoised columns.
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        denoised = denoise_series(df[col].values, method=method, **kwargs)
        result[f"{prefix}{col}"] = denoised

    return result
