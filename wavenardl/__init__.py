"""
wavenardl — Wavelet-based Nonlinear ARDL (W-NARDL) for Python
==============================================================

A comprehensive Python library implementing:

1. **NARDL** — Nonlinear Autoregressive Distributed Lag models
   (Shin, Yu & Greenwood-Nimmo, 2014)

2. **Wavelet NARDL** — Wavelet-denoised NARDL using Haar à Trous
   wavelet transform (Jammazi, Lahiani & Nguyen, 2015)

Key Features
------------
- Full NARDL estimation with asymmetric decomposition
- Haar à Trous wavelet denoising (paper method)
- PyWavelets integration (SWT, DWT alternatives)
- Automatic lag selection (AIC, BIC, AICc, HQ)
- PSS bounds F-test for cointegration
- Narayan small-sample test
- Wald symmetry tests (long-run and short-run)
- Dynamic multipliers with bootstrap confidence intervals
- Long-run multipliers via delta method
- Error Correction Model (ECM) representation
- Comprehensive diagnostic tests (BG, BP, JB, RESET, CUSUM)
- Publication-quality tables (LaTeX, HTML, console)
- Beautiful visualizations (scalograms, multiplier plots, diagnostics)

Quick Start
-----------
>>> from wavenardl import NARDL, WaveletNARDL
>>> import pandas as pd
>>>
>>> data = pd.read_csv('my_data.csv')
>>>
>>> # Standard NARDL
>>> model = NARDL(data, "y ~ x1 + x2 + Asymmetric(x1)", maxlag=4)
>>> results = model.fit()
>>> results.summary()
>>>
>>> # Wavelet NARDL
>>> wmodel = WaveletNARDL(data, "y ~ x1 + Asymmetric(x1)", wavelet_method='htw')
>>> wresults = wmodel.fit()
>>> wresults['comparison'].summary()

References
----------
Jammazi, R., Lahiani, A., & Nguyen, D. K. (2015). A wavelet-based nonlinear
    ARDL model for assessing the exchange rate pass-through to crude oil prices.
    Journal of International Financial Markets, Institutions and Money, 34, 173-187.

Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014). Modelling asymmetric
    cointegration and dynamic multipliers in a nonlinear ARDL framework.
    Festschrift in Honor of Peter Schmidt, 281-314.

Pesaran, M. H., Shin, Y., & Smith, R. (2001). Bounds testing approaches
    to the analysis of level relationship. JASA, 16(3), 289-326.

Narayan, P. K. (2005). The saving and investment nexus for China.
    Applied Economics, 37(17), 1979-1990.
"""

__version__ = '1.0.1'
__author__ = 'Dr Merwan Roudane'

# ── Core Classes ──────────────────────────────────────────────────────
from .nardl import NARDL, NARDLResults
from .wavenardl import WaveletNARDL, WNARDLComparison

# ── Wavelet Module ────────────────────────────────────────────────────
from .wavelet import (
    HaarATrousWavelet,
    PyWaveletDenoiser,
    denoise_series,
    denoise_dataframe,
)

# ── Data Preparation ─────────────────────────────────────────────────
from .prepare import (
    partial_sum_decomposition,
    prepare_nardl_data,
)

# ── Statistical Tests ────────────────────────────────────────────────
from .tests import pss_f_test, symmetry_test, TestResult

# ── Long-run & Multipliers ───────────────────────────────────────────
from .longrun import compute_long_run_multipliers, long_run_summary
from .multipliers import DynamicMultipliers, bootstrap_multipliers

# ── ECM ──────────────────────────────────────────────────────────────
from .ecm import estimate_ecm, ECMResults

# ── Diagnostics ──────────────────────────────────────────────────────
from .diagnostics import (
    run_all_diagnostics,
    cusum_test,
    cusum_sq_test,
    diagnostics_summary,
)

# ── Visualization ────────────────────────────────────────────────────
from .visualize import (
    plot_wavelet_decomposition,
    plot_scalogram,
    plot_multipliers,
    plot_residual_diagnostics,
    plot_cusum,
    plot_lag_criteria,
    plot_coefficient_comparison,
)

# ── Tables ───────────────────────────────────────────────────────────
from .tables import (
    summary_table,
    comparison_table,
    bounds_test_table,
    diagnostics_table,
)

# ── Lag Selection ────────────────────────────────────────────────────
from .lagselect import grid_search, quick_search, select_lags

# ── Utilities ────────────────────────────────────────────────────────
from .utils import FormulaParser, compute_ic

__all__ = [
    # Core
    'NARDL', 'NARDLResults', 'WaveletNARDL', 'WNARDLComparison',
    # Wavelet
    'HaarATrousWavelet', 'PyWaveletDenoiser',
    'denoise_series', 'denoise_dataframe',
    # Data
    'partial_sum_decomposition', 'prepare_nardl_data',
    # Tests
    'pss_f_test', 'symmetry_test', 'TestResult',
    # Long-run & Multipliers
    'compute_long_run_multipliers', 'long_run_summary',
    'DynamicMultipliers', 'bootstrap_multipliers',
    # ECM
    'estimate_ecm', 'ECMResults',
    # Diagnostics
    'run_all_diagnostics', 'cusum_test', 'cusum_sq_test', 'diagnostics_summary',
    # Visualization
    'plot_wavelet_decomposition', 'plot_scalogram',
    'plot_multipliers', 'plot_residual_diagnostics',
    'plot_cusum', 'plot_lag_criteria', 'plot_coefficient_comparison',
    # Tables
    'summary_table', 'comparison_table', 'bounds_test_table', 'diagnostics_table',
    # Lag Selection
    'grid_search', 'quick_search', 'select_lags',
    # Utilities
    'FormulaParser', 'compute_ic',
]
