"""Generate all documentation images for the wavenardl docs site."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats

from wavenardl import (
    NARDL, WaveletNARDL,
    HaarATrousWavelet, denoise_series,
    partial_sum_decomposition,
    pss_f_test, symmetry_test,
    compute_long_run_multipliers,
    DynamicMultipliers, bootstrap_multipliers,
    run_all_diagnostics, cusum_test, cusum_sq_test,
    plot_wavelet_decomposition, plot_multipliers,
    plot_residual_diagnostics, plot_cusum,
    plot_lag_criteria, plot_coefficient_comparison,
)

IMG = os.path.join(os.path.dirname(__file__), 'images')

P = {
    'blue': '#2563EB', 'red': '#DC2626', 'green': '#059669',
    'purple': '#7C3AED', 'amber': '#D97706', 'cyan': '#0891B2',
    'indigo': '#4F46E5', 'emerald': '#10B981', 'gray': '#6B7280',
    'dark': '#1F2937', 'rose': '#E11D48',
}

plt.rcParams.update({
    'figure.dpi': 150, 'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial'], 'figure.facecolor': 'white',
    'axes.titlesize': 13, 'axes.labelsize': 11,
})

np.random.seed(2024)
n = 280
dates = pd.date_range('2000-01-01', periods=n, freq='MS')

exrate = np.zeros(n); exrate[0] = np.log(0.92)
for t in range(1, n):
    m = np.log(1.25) if 96 <= t <= 180 else np.log(1.10)
    exrate[t] = exrate[t-1] + 0.03*(m - exrate[t-1]) + np.random.normal(0, 0.018)
exrate = np.exp(exrate)

cpi = np.zeros(n); cpi[0] = 170
for t in range(1, n):
    inf = 0.006 if 240 <= t <= 260 else 0.002
    cpi[t] = cpi[t-1] * (1 + inf + np.random.normal(0, 0.001))

oil = np.zeros(n); oil[0] = np.log(28)
for t in range(1, n):
    dx = exrate[t] - exrate[t-1]
    pt = 3.5*dx if dx > 0 else 1.2*dx
    ce = 0.15*(np.log(cpi[t]) - np.log(cpi[t-1]))
    shock = 0
    if 100 <= t <= 108: shock = np.random.normal(0, 0.06)
    elif 240 <= t <= 244: shock = np.random.normal(-0.08, 0.05)
    oil[t] = oil[t-1] + 0.005 + pt + ce + shock + np.random.normal(0, 0.04)
oil = np.exp(oil)

crisis = np.zeros(n); crisis[96:114] = 1
data = pd.DataFrame({'oil': oil, 'exrate': exrate, 'cpi': cpi, 'crisis': crisis}, index=dates)
formula = "oil ~ exrate + cpi + Asymmetric(exrate) + deterministic(crisis)"

print("Generating images...")

# 1. Hero banner data plot
fig = plt.figure(figsize=(14, 4.5), facecolor='white')
ax = fig.add_subplot(111)
ax.fill_between(data.index, data['oil'], alpha=0.10, color=P['blue'])
ax.plot(data.index, data['oil'], color=P['blue'], lw=1.8, label='Brent Oil')
ax.plot(data.index, data['exrate']*80, color=P['emerald'], lw=1.2, alpha=0.7, label='EUR/USD (×80)')
cm = data['crisis'] == 1
if cm.any(): ax.axvspan(data.index[cm][0], data.index[cm][-1], alpha=0.06, color=P['red'])
ax.set_ylabel('USD/barrel', fontsize=11)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.15, linestyle='--')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.set_title('Oil Prices & Exchange Rate — Monthly Data (2000–2023)', fontsize=14, fontweight='bold')
plt.tight_layout(); fig.savefig(f'{IMG}/hero_timeseries.png', dpi=150, bbox_inches='tight'); plt.close()
print("  1/12 hero_timeseries")

# 2. EDA panel
fig = plt.figure(figsize=(14, 8), facecolor='white')
gs = GridSpec(2, 2, hspace=0.35, wspace=0.25)
ax1 = fig.add_subplot(gs[0, :])
ax1.fill_between(data.index, data['oil'], alpha=0.12, color=P['blue'])
ax1.plot(data.index, data['oil'], color=P['blue'], lw=1.8)
if cm.any(): ax1.axvspan(data.index[cm][0], data.index[cm][-1], alpha=0.06, color=P['red'], label='Crisis')
ax1.set_title('Panel A: Brent Crude Oil Prices', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.15, linestyle='--')
for sp in ['top','right']: ax1.spines[sp].set_visible(False)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(data.index, data['exrate'], color=P['emerald'], lw=1.5)
ax2.set_title('Panel B: EUR/USD', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.15, linestyle='--')
for sp in ['top','right']: ax2.spines[sp].set_visible(False)

ax3 = fig.add_subplot(gs[1, 1])
d_o = np.diff(np.log(data['oil'])); d_e = np.diff(np.log(data['exrate']))
pm = d_e > 0; nm = d_e <= 0
ax3.scatter(d_e[pm], d_o[pm], c=P['blue'], alpha=0.4, s=18, label='ExRate ↑')
ax3.scatter(d_e[nm], d_o[nm], c=P['red'], alpha=0.4, s=18, label='ExRate ↓')
for mask, clr in [(pm, P['blue']), (nm, P['red'])]:
    if np.sum(mask) > 2:
        z = np.polyfit(d_e[mask], d_o[mask], 1); p = np.poly1d(z)
        xf = np.linspace(d_e[mask].min(), d_e[mask].max(), 50)
        ax3.plot(xf, p(xf), color=clr, lw=2, ls='--')
ax3.axhline(0, color=P['gray'], lw=0.5); ax3.axvline(0, color=P['gray'], lw=0.5)
ax3.set_title('Panel C: Asymmetric Co-movement', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.15, linestyle='--')
for sp in ['top','right']: ax3.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(f'{IMG}/eda_panel.png', dpi=150, bbox_inches='tight'); plt.close()
print("  2/12 eda_panel")

# 3. Wavelet decomposition
htw = HaarATrousWavelet(n_levels=5, threshold_method='soft')
oil_analysis = htw.full_analysis(data['oil'].values, name='Brent Oil')
fig = plot_wavelet_decomposition(oil_analysis, figsize=(14, 12))
fig.suptitle('HTW Wavelet Decomposition — Brent Oil', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(); fig.savefig(f'{IMG}/wavelet_decomp.png', dpi=150, bbox_inches='tight'); plt.close()
print("  3/12 wavelet_decomp")

# 4. Denoising comparison
dn_htw = denoise_series(data['oil'].values, method='htw', n_levels=5)
dn_swt = denoise_series(data['oil'].values, method='swt', n_levels=5)
fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
ax.plot(data.index, data['oil'], color=P['gray'], alpha=0.4, lw=0.8, label='Original')
ax.plot(data.index, dn_htw, color=P['blue'], lw=2, label='HTW (Paper)')
ax.plot(data.index, dn_swt, color=P['emerald'], lw=1.5, ls='--', label='SWT (PyWavelets)')
ax.set_title('Wavelet Denoising Methods Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.15, linestyle='--')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(f'{IMG}/denoise_compare.png', dpi=150, bbox_inches='tight'); plt.close()
print("  4/12 denoise_compare")

# 5. Partial sum decomposition
ep, en = partial_sum_decomposition(data['exrate'].values)
fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
ax2t = ax.twinx()
l1 = ax.plot(data.index, data['exrate'], color=P['dark'], lw=1.5, alpha=0.6, label='ExRate')
l2 = ax2t.plot(data.index, ep, color=P['blue'], lw=2, label='x⁺ (Positive)')
l3 = ax2t.plot(data.index, en, color=P['red'], lw=2, label='x⁻ (Negative)')
lines = l1+l2+l3; ax.legend(lines, [l.get_label() for l in lines], fontsize=10, loc='upper left')
ax.set_title('Partial Sum Decomposition — EUR/USD', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.15, linestyle='--')
plt.tight_layout(); fig.savefig(f'{IMG}/partial_sums.png', dpi=150, bbox_inches='tight'); plt.close()
print("  5/12 partial_sums")

# 6. NARDL estimation
model = NARDL(data=data, formula=formula, maxlag=4, criterion='BIC', case=3)
results = model.fit(mode='grid', verbose=False)

coef = results.coef_table
fig, ax = plt.subplots(figsize=(11, max(5, len(coef)*0.38)), facecolor='white')
vals = coef['Coefficient'].values; errs = coef['Std.Error'].values*1.96; pvs = coef['p-value'].values
clrs = [P['blue'] if p<0.01 else P['indigo'] if p<0.05 else P['amber'] if p<0.1 else P['gray'] for p in pvs]
ax.barh(range(len(coef)), vals, xerr=errs, color=clrs, alpha=0.75, edgecolor='white', capsize=3)
ax.axvline(0, color=P['dark'], lw=0.8); ax.set_yticks(range(len(coef)))
ax.set_yticklabels(coef.index, fontsize=9); ax.invert_yaxis()
ax.set_title('NARDL Coefficient Estimates with 95% CI', fontsize=14, fontweight='bold')
le = [Patch(facecolor=P['blue'], alpha=0.75, label='p<0.01 ***'),
      Patch(facecolor=P['indigo'], alpha=0.75, label='p<0.05 **'),
      Patch(facecolor=P['amber'], alpha=0.75, label='p<0.10 *'),
      Patch(facecolor=P['gray'], alpha=0.75, label='p≥0.10')]
ax.legend(handles=le, fontsize=8, loc='lower right'); ax.grid(True, alpha=0.15, linestyle='--', axis='x')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(f'{IMG}/nardl_coefs.png', dpi=150, bbox_inches='tight'); plt.close()
print("  6/12 nardl_coefs")

# 7. PSS bounds test
pss = pss_f_test(results, case=3)
cv = pss.details.get('Critical Values', pd.DataFrame())
if not cv.empty:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    sl = list(cv.index); i0 = cv['I(0)'].values; i1 = cv['I(1)'].values; xp = np.arange(len(sl)); w=0.35
    ax.bar(xp-w/2, i0, w, color=P['blue'], alpha=0.7, label='I(0) bound')
    ax.bar(xp+w/2, i1, w, color=P['red'], alpha=0.7, label='I(1) bound')
    ax.axhline(pss.statistic, color=P['emerald'], lw=2.5, ls='--', label=f'F = {pss.statistic:.3f}')
    ax.set_xticks(xp); ax.set_xticklabels(sl, fontsize=11)
    ax.set_title('PSS Bounds F-Test for Cointegration', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.15, linestyle='--', axis='y')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    plt.tight_layout(); fig.savefig(f'{IMG}/pss_bounds.png', dpi=150, bbox_inches='tight'); plt.close()
print("  7/12 pss_bounds")

# 8. Long-run multipliers
lr = compute_long_run_multipliers(results)
fig, ax = plt.subplots(figsize=(10, max(3.5, len(lr)*1.1)), facecolor='white')
clrs_lr = [P['blue'] if 'POS' in str(n) else P['red'] if 'NEG' in str(n) else P['purple'] for n in lr.index]
ax.barh(range(len(lr)), lr['LR_Coefficient'].values, xerr=lr['Std.Error'].values*1.96,
        color=clrs_lr, alpha=0.75, capsize=4)
ax.axvline(0, color=P['dark'], lw=0.8); ax.set_yticks(range(len(lr)))
ax.set_yticklabels(lr.index, fontsize=11); ax.invert_yaxis()
ax.set_title('Long-Run Multipliers (Delta Method) with 95% CI', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.15, linestyle='--', axis='x')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(f'{IMG}/longrun_mult.png', dpi=150, bbox_inches='tight'); plt.close()
print("  8/12 longrun_mult")

# 9. Dynamic multipliers
dm = DynamicMultipliers(results, horizon=80)
fig = plot_multipliers(dm.mpsi, dep_var='oil', figsize=(13, 5.5))
fig.suptitle('Cumulative Dynamic Multipliers', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(f'{IMG}/dynamic_mult.png', dpi=150, bbox_inches='tight'); plt.close()
print("  9/12 dynamic_mult")

# 10. Residual diagnostics
fig = plot_residual_diagnostics(results, figsize=(14, 10))
fig.suptitle('Residual Diagnostics Panel', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(f'{IMG}/residual_diag.png', dpi=150, bbox_inches='tight'); plt.close()
print("  10/12 residual_diag")

# 11. CUSUM
fig = plot_cusum(results, figsize=(14, 4.5))
fig.suptitle('CUSUM & CUSUMSQ Stability Tests', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout(); fig.savefig(f'{IMG}/cusum.png', dpi=150, bbox_inches='tight'); plt.close()
print("  11/12 cusum")

# 12. W-NARDL comparison
wm = WaveletNARDL(data=data, formula=formula, maxlag=4, wavelet_method='htw', n_levels=5)
wr = wm.fit(mode='grid', verbose=False, fit_original=True)
fig = plot_coefficient_comparison(wr['original'], wr['wavelet'], figsize=(15, max(5, results.nparams*0.32)))
fig.suptitle('Coefficient Comparison: Standard NARDL vs Wavelet NARDL', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(f'{IMG}/wnardl_compare.png', dpi=150, bbox_inches='tight'); plt.close()

# W-NARDL dashboard
o = wr['original']; w = wr['wavelet']
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='white')
for ax, metric, ov, wv in [(axes[0],'R²',o.rsquared,w.rsquared),
                             (axes[1],'AIC (norm)',o.aic_norm,w.aic_norm),
                             (axes[2],'BIC (norm)',o.bic_norm,w.bic_norm)]:
    bars = ax.bar(['Standard\nNARDL','Wavelet\nNARDL'], [ov,wv],
                  color=[P['blue'],P['emerald']], alpha=0.8, width=0.5)
    for b,v in zip(bars,[ov,wv]):
        ax.text(b.get_x()+b.get_width()/2, v+abs(v)*0.005, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.15, linestyle='--', axis='y')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(f'{IMG}/wnardl_dashboard.png', dpi=150, bbox_inches='tight'); plt.close()
print("  12/12 wnardl_compare + dashboard")

print(f"\n✅ All 12 images saved to {IMG}/")
