"""
Test script for the wavenardl package.

Creates synthetic data and tests all major components.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
np.random.seed(42)

print("=" * 70)
print("  WAVENARDL PACKAGE - COMPREHENSIVE TEST SUITE")
print("=" * 70)

# -- Generate Synthetic Data -----------------------------------------------
print("\n[1/10] Generating synthetic data...")

n = 200
t = np.arange(n)
eps = np.random.normal(0, 0.5, n)
x = np.cumsum(np.random.normal(0.02, 0.5, n))

y = np.zeros(n)
y[0] = 10
for i in range(1, n):
    dx = x[i] - x[i-1]
    if dx > 0:
        effect = 0.8 * dx
    else:
        effect = 0.3 * dx
    y[i] = 0.95 * y[i-1] + effect + 0.1 * np.sin(0.1*i) + eps[i]

z = np.cumsum(np.random.normal(0, 0.3, n)) + 5
dummy = np.zeros(n)
dummy[100:120] = 1

data = pd.DataFrame({
    'oil': y,
    'exrate': x,
    'cpi': z,
    'crisis': dummy,
})

print(f"  Data shape: {data.shape}")
print(f"  Variables: {list(data.columns)}")
print("  OK: Synthetic data generated.")

# -- Test Partial Sum Decomposition ----------------------------------------
print("\n[2/10] Testing partial sum decomposition...")
from wavenardl.prepare import partial_sum_decomposition

x_pos, x_neg = partial_sum_decomposition(data['exrate'].values)
print(f"  X_POS range: [{np.nanmin(x_pos):.4f}, {np.nanmax(x_pos):.4f}]")
print(f"  X_NEG range: [{np.nanmin(x_neg):.4f}, {np.nanmax(x_neg):.4f}]")
assert not np.any(x_pos[~np.isnan(x_pos)] < 0), "X_POS should be non-negative"
assert not np.any(x_neg[~np.isnan(x_neg)] > 0), "X_NEG should be non-positive"
print("  OK: Partial sum decomposition correct.")

# -- Test Wavelet Denoising (HTW) -----------------------------------------
print("\n[3/10] Testing Haar a Trous Wavelet denoising...")
from wavenardl.wavelet import HaarATrousWavelet, PyWaveletDenoiser, denoise_series

htw = HaarATrousWavelet(n_levels=4, threshold_method='soft')
analysis = htw.full_analysis(data['oil'].values, name='oil')

print(f"  N levels     : {analysis['n_levels']}")
print(f"  Threshold    : {analysis['threshold']:.4f}")
print(f"  Sigma (noise): {analysis['sigma']:.4f}")
print(f"  Original std : {np.std(analysis['original']):.4f}")
print(f"  Denoised std : {np.std(analysis['denoised']):.4f}")
assert len(analysis['details']) == 4, "Should have 4 detail levels"
print("  OK: HTW wavelet denoising works.")

# Test PyWavelets
print("\n[4/10] Testing PyWavelets denoising (SWT)...")
pywt_denoiser = PyWaveletDenoiser(wavelet='haar', method='swt', n_levels=4)
pywt_analysis = pywt_denoiser.full_analysis(data['oil'].values, name='oil')
print(f"  PyWavelets method: {pywt_analysis['method']}")
print(f"  Wavelet          : {pywt_analysis['wavelet']}")
print(f"  Denoised std     : {np.std(pywt_analysis['denoised']):.4f}")

denoised_htw = denoise_series(data['oil'].values, method='htw')
denoised_swt = denoise_series(data['oil'].values, method='swt')
print(f"  HTW denoised std : {np.std(denoised_htw):.4f}")
print(f"  SWT denoised std : {np.std(denoised_swt):.4f}")
print("  OK: PyWavelets integration works.")

# -- Test NARDL Estimation -------------------------------------------------
print("\n[5/10] Testing NARDL estimation...")
from wavenardl import NARDL

model = NARDL(
    data=data,
    formula="oil ~ exrate + cpi + Asymmetric(exrate) + deterministic(crisis)",
    maxlag=3,
    criterion='BIC',
    case=3,
)

results = model.fit(mode='quick', verbose=False)
print(f"  Observations    : {results.nobs}")
print(f"  Parameters      : {results.nparams}")
print(f"  R-squared       : {results.rsquared:.4f}")
print(f"  Adj. R-squared  : {results.rsquared_adj:.4f}")
print(f"  AIC             : {results.aic:.4f}")
print(f"  BIC             : {results.bic:.4f}")
print(f"  Optimal lags    : {results.optimal_lags}")
print(f"  DW              : {results.durbin_watson:.4f}")

results.summary()
print("  OK: NARDL estimation successful.")

# -- Test PSS Bounds Test --------------------------------------------------
print("\n[6/10] Testing PSS bounds F-test...")
from wavenardl import pss_f_test

pss_result = pss_f_test(results, case=3)
print(f"  F-statistic: {pss_result.statistic:.4f}")
print(f"  Decision: {pss_result.decision}")
print("  OK: PSS bounds test works.")

# -- Test Symmetry Test ----------------------------------------------------
print("\n[7/10] Testing symmetry Wald tests...")
from wavenardl import symmetry_test

sym_result = symmetry_test(results)
print(f"  LR tests: {sym_result.statistic.get('n_LR_tests', 0)}")
print(f"  SR tests: {sym_result.statistic.get('n_SR_tests', 0)}")
print("  OK: Symmetry tests work.")

# -- Test Long-Run Multipliers ---------------------------------------------
print("\n[8/10] Testing long-run multipliers...")
from wavenardl import compute_long_run_multipliers, long_run_summary

lr = compute_long_run_multipliers(results)
print(lr.to_string())
long_run_summary(results)
print("  OK: Long-run multipliers computed.")

# -- Test Dynamic Multipliers ----------------------------------------------
print("\n[9/10] Testing dynamic multipliers...")
from wavenardl import DynamicMultipliers

dm = DynamicMultipliers(results, horizon=40)
dm.summary()
print(f"  Multiplier DataFrame shape: {dm.mpsi.shape}")
print("  OK: Dynamic multipliers computed.")

# -- Test Diagnostics ------------------------------------------------------
print("\n[10/10] Testing diagnostics...")
from wavenardl import run_all_diagnostics, diagnostics_summary

diag = run_all_diagnostics(results)
print(diag.to_string())
diagnostics_summary(results)
print("  OK: Diagnostics complete.")

# -- Test W-NARDL ----------------------------------------------------------
print("\n[BONUS] Testing Wavelet NARDL (full pipeline)...")
from wavenardl import WaveletNARDL

wmodel = WaveletNARDL(
    data=data,
    formula="oil ~ exrate + cpi + Asymmetric(exrate) + deterministic(crisis)",
    maxlag=3,
    wavelet_method='htw',
    n_levels=4,
)

wresults = wmodel.fit(mode='quick', verbose=False, fit_original=True)
print(f"\n  W-NARDL Results:")
print(f"    R2 (Original)  : {wresults['original'].rsquared:.4f}")
print(f"    R2 (Wavelet)   : {wresults['wavelet'].rsquared:.4f}")
print(f"    BIC (Original) : {wresults['original'].bic:.4f}")
print(f"    BIC (Wavelet)  : {wresults['wavelet'].bic:.4f}")

wresults['comparison'].summary()

# Test table exports
print("\n  Testing LaTeX export...")
from wavenardl import summary_table
latex = summary_table(results, format='latex')
print(f"  LaTeX output: {len(latex)} characters")

html = summary_table(results, format='html')
print(f"  HTML output:  {len(html)} characters")

print("\n" + "=" * 70)
print("  ALL TESTS PASSED SUCCESSFULLY")
print("=" * 70)
