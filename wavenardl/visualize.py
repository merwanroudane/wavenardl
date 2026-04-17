"""
Beautiful visualizations for Wavelet-NARDL models.

Premium, publication-quality plots with modern aesthetics:
- Wavelet decomposition & denoising (paper Fig. 1 style)
- Dynamic multiplier plots with bootstrap CI
- Residual diagnostics panel
- Coefficient comparison charts
- Lag selection criteria plots
- CUSUM / CUSUMSQ stability plots
- Wavelet scalogram heatmaps

Color schemes inspired by the paper and modern data visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Optional, Tuple
import warnings

# ── Theme Configuration ──────────────────────────────────────────────

# Premium color palette
COLORS = {
    'primary': '#2563EB',        # Vibrant blue
    'secondary': '#DC2626',      # Rich red
    'accent': '#059669',         # Emerald green
    'positive': '#3B82F6',       # Bright blue
    'negative': '#EF4444',       # Bright red
    'asymmetry': '#1F2937',      # Dark charcoal
    'ci_fill': '#93C5FD',        # Light blue
    'ci_alpha': 0.25,
    'grid': '#E5E7EB',           # Light gray
    'bg': '#FAFBFC',             # Off-white
    'text': '#1F2937',           # Dark text
    'muted': '#9CA3AF',          # Muted gray
    'wavelet_original': '#DC2626',  # Red (original, as in paper)
    'wavelet_denoised': '#2563EB',  # Blue (smoothed, as in paper)
    'gradient_start': '#667EEA',
    'gradient_end': '#764BA2',
}

# Wavelet detail level colors (beautiful gradient)
WAVELET_COLORS = [
    '#FF6B6B', '#FFA07A', '#FFD700', '#98FB98',
    '#87CEEB', '#9370DB', '#FF69B4', '#40E0D0',
]


def _setup_style(ax, title='', xlabel='', ylabel='', dark_mode=False):
    """Apply consistent premium styling to axes."""
    if dark_mode:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#e0e0e0')
        ax.xaxis.label.set_color('#e0e0e0')
        ax.yaxis.label.set_color('#e0e0e0')
        ax.title.set_color('#ffffff')
        for spine in ax.spines.values():
            spine.set_color('#333355')
    else:
        ax.set_facecolor(COLORS['bg'])
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(COLORS['grid'])

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12,
                 color=COLORS['text'] if not dark_mode else '#fff')
    ax.set_xlabel(xlabel, fontsize=10, color=COLORS['muted'])
    ax.set_ylabel(ylabel, fontsize=10, color=COLORS['muted'])
    ax.grid(True, alpha=0.3, linestyle='--', color=COLORS['grid'])
    ax.tick_params(labelsize=9)


# ══════════════════════════════════════════════════════════════════════
# Wavelet Decomposition Plots
# ══════════════════════════════════════════════════════════════════════

def plot_wavelet_decomposition(
    analysis: Dict,
    figsize: Tuple = (14, 10),
    dark_mode: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot wavelet decomposition: original vs denoised + detail levels.

    Matches the style of Fig. 1 in Jammazi et al. (2015):
    smoothed (blue) vs original (red).
    """
    name = analysis.get('name', 'Signal')
    original = analysis['original']
    denoised = analysis['denoised']
    details = analysis.get('details', analysis.get('decomposition', {}).get('details', []))

    n_details = len(details) if details else 0
    n_plots = 2 + n_details  # original+denoised, smooth, each detail

    fig = plt.figure(figsize=figsize, facecolor='#0d1117' if dark_mode else 'white')
    gs = GridSpec(n_plots, 1, hspace=0.4)

    t = np.arange(len(original))

    # Main plot: Original vs Denoised
    ax0 = fig.add_subplot(gs[0:2, 0])
    ax0.plot(t, original, color=COLORS['wavelet_original'],
             alpha=0.6, linewidth=0.8, label='Original')
    ax0.plot(t, denoised, color=COLORS['wavelet_denoised'],
             linewidth=1.5, label='Denoised (Smoothed)')
    _setup_style(ax0, f'HTW Decomposition — {name}',
                 'Time', 'Value', dark_mode)
    ax0.legend(fontsize=9, loc='upper left',
               framealpha=0.8, edgecolor='none')

    # Detail levels
    if details:
        for j, d in enumerate(details):
            ax = fig.add_subplot(gs[2 + j, 0])
            color = WAVELET_COLORS[j % len(WAVELET_COLORS)]
            ax.plot(t[:len(d)], d[:len(t)], color=color,
                    linewidth=0.7, alpha=0.9)
            ax.axhline(y=0, color=COLORS['muted'], linewidth=0.5, linestyle='--')
            _setup_style(ax, f'Detail d{j+1}', '', '', dark_mode)
            ax.set_ylabel(f'd{j+1}', fontsize=9, rotation=0, labelpad=20)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    return fig


def plot_scalogram(
    analysis: Dict,
    figsize: Tuple = (14, 5),
    cmap: str = 'magma',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot wavelet scalogram (power spectrum heatmap).

    Beautiful heatmap showing wavelet coefficient magnitudes
    across time and scale.
    """
    details = analysis.get('details', [])
    if not details:
        warnings.warn("No detail coefficients for scalogram.")
        return None

    name = analysis.get('name', 'Signal')

    # Build 2D matrix of |detail coefficients|
    max_len = max(len(d) for d in details)
    n_levels = len(details)

    power = np.zeros((n_levels, max_len))
    for j, d in enumerate(details):
        power[j, :len(d)] = np.abs(d[:max_len]) ** 2

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    im = ax.imshow(power, aspect='auto', cmap=cmap,
                   interpolation='bilinear',
                   extent=[0, max_len, n_levels + 0.5, 0.5])

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Power |d_j|²', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_ylabel('Scale (level)', fontsize=11)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_title(f'Wavelet Scalogram — {name}', fontsize=13,
                 fontweight='bold', pad=12)
    ax.set_yticks(range(1, n_levels + 1))
    ax.set_yticklabels([f'd{j}' for j in range(1, n_levels + 1)])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ══════════════════════════════════════════════════════════════════════
# Dynamic Multiplier Plots
# ══════════════════════════════════════════════════════════════════════

def plot_multipliers(
    mpsi: pd.DataFrame,
    variable: Optional[str] = None,
    dep_var: str = 'y',
    figsize: Tuple = (12, 6),
    dark_mode: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot dynamic multipliers with bootstrap confidence intervals.

    Shows positive change, negative change, and asymmetry curves
    with (optional) bootstrap CI bands.
    """
    from .utils import pos_name, neg_name

    # Identify variables to plot
    if variable:
        vars_to_plot = [variable]
    else:
        # Find all unique variable names from columns
        vars_to_plot = []
        for col in mpsi.columns:
            if col.endswith('_diff'):
                vars_to_plot.append(col.replace('_diff', ''))

    n_vars = len(vars_to_plot)
    fig, axes = plt.subplots(1, n_vars, figsize=(figsize[0], figsize[1]),
                             squeeze=False,
                             facecolor='#0d1117' if dark_mode else 'white')

    for i, var in enumerate(vars_to_plot):
        ax = axes[0, i]
        h = mpsi['h'].values

        pn = pos_name(var) if pos_name(var) in mpsi.columns else var
        nn = neg_name(var) if neg_name(var) in mpsi.columns else var
        diff_col = f"{var}_diff"

        # Plot positive and negative multipliers
        if pn in mpsi.columns:
            ax.plot(h, mpsi[pn].values, color=COLORS['positive'],
                    linewidth=2, label='Positive Change', zorder=3)
        if nn in mpsi.columns:
            ax.plot(h, mpsi[nn].values, color=COLORS['negative'],
                    linewidth=2, label='Negative Change', zorder=3)

        # Plot asymmetry
        if diff_col in mpsi.columns:
            ax.plot(h, mpsi[diff_col].values, color=COLORS['asymmetry'],
                    linewidth=1.5, linestyle='--', label='Asymmetry', zorder=4)

        # Bootstrap CI
        uci_col = f"{var}_uCI"
        lci_col = f"{var}_lCI"
        if uci_col in mpsi.columns and lci_col in mpsi.columns:
            ax.fill_between(h, mpsi[lci_col].values, mpsi[uci_col].values,
                            color=COLORS['ci_fill'], alpha=COLORS['ci_alpha'],
                            label='95% CI', zorder=1)

        ax.axhline(y=0, color=COLORS['muted'], linewidth=0.8, linestyle='-', zorder=2)

        _setup_style(ax,
                     f'Cumulative Effect of {var} on {dep_var}',
                     'Horizon (h)', 'Multiplier', dark_mode)
        ax.legend(fontsize=8, loc='best', framealpha=0.8, edgecolor='none')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    return fig


# ══════════════════════════════════════════════════════════════════════
# Residual Diagnostics Panel
# ══════════════════════════════════════════════════════════════════════

def plot_residual_diagnostics(
    nardl_results,
    figsize: Tuple = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel residual diagnostic plot:
    1. Residuals vs Time
    2. Histogram + KDE
    3. Q-Q Plot
    4. ACF Plot
    """
    from scipy import stats as sp_stats
    from statsmodels.graphics.tsaplots import plot_acf

    resid = nardl_results.residuals.values
    fitted = nardl_results.fitted_values.values

    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='white')

    # 1. Residuals vs Time
    ax = axes[0, 0]
    ax.plot(resid, color=COLORS['primary'], linewidth=0.8, alpha=0.8)
    ax.axhline(y=0, color=COLORS['secondary'], linewidth=1, linestyle='--')
    ax.fill_between(range(len(resid)), resid, 0,
                    where=resid > 0, color=COLORS['positive'], alpha=0.15)
    ax.fill_between(range(len(resid)), resid, 0,
                    where=resid < 0, color=COLORS['negative'], alpha=0.15)
    _setup_style(ax, 'Residuals over Time', 'Observation', 'Residual')

    # 2. Histogram + KDE
    ax = axes[0, 1]
    ax.hist(resid, bins=30, density=True, color=COLORS['primary'],
            alpha=0.5, edgecolor='white', linewidth=0.5)
    x_range = np.linspace(resid.min(), resid.max(), 200)
    kde = sp_stats.gaussian_kde(resid)
    ax.plot(x_range, kde(x_range), color=COLORS['secondary'],
            linewidth=2, label='KDE')
    # Normal overlay
    ax.plot(x_range, sp_stats.norm.pdf(x_range, resid.mean(), resid.std()),
            color=COLORS['accent'], linewidth=1.5, linestyle='--',
            label='Normal')
    ax.legend(fontsize=8, framealpha=0.8)
    _setup_style(ax, 'Residual Distribution', 'Value', 'Density')

    # 3. Q-Q Plot
    ax = axes[1, 0]
    sp_stats.probplot(resid, dist='norm', plot=ax)
    ax.get_lines()[0].set(color=COLORS['primary'], markersize=3, alpha=0.6)
    ax.get_lines()[1].set(color=COLORS['secondary'], linewidth=1.5)
    _setup_style(ax, 'Q-Q Plot (Normal)', 'Theoretical Quantiles',
                 'Sample Quantiles')

    # 4. Correlogram
    ax = axes[1, 1]
    plot_acf(resid, ax=ax, lags=20, alpha=0.05,
             color=COLORS['primary'], vlines_kwargs={'colors': COLORS['primary']})
    _setup_style(ax, 'Autocorrelation Function', 'Lag', 'ACF')

    fig.suptitle('Residual Diagnostics', fontsize=15, fontweight='bold',
                 y=1.02, color=COLORS['text'])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ══════════════════════════════════════════════════════════════════════
# CUSUM Plot
# ══════════════════════════════════════════════════════════════════════

def plot_cusum(
    nardl_results,
    figsize: Tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot CUSUM and CUSUMSQ stability tests."""
    from .diagnostics import cusum_test, cusum_sq_test

    cusum = cusum_test(nardl_results)
    cusumsq = cusum_sq_test(nardl_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')

    # CUSUM
    t = cusum['t_values'][:len(cusum['cusum'])]
    ax1.plot(t, cusum['cusum'], color=COLORS['primary'], linewidth=1.5)
    ax1.plot(t, cusum['upper'][:len(t)], color=COLORS['secondary'],
             linewidth=1, linestyle='--', label='5% bounds')
    ax1.plot(t, cusum['lower'][:len(t)], color=COLORS['secondary'],
             linewidth=1, linestyle='--')
    ax1.fill_between(t, cusum['lower'][:len(t)], cusum['upper'][:len(t)],
                     color=COLORS['accent'], alpha=0.08)
    _setup_style(ax1, f'CUSUM Test ({"Stable ✓" if cusum["stable"] else "Unstable ✗"})',
                 'Observation', 'CUSUM')
    ax1.legend(fontsize=8)

    # CUSUMSQ
    t2 = cusumsq['t_values']
    ax2.plot(t2, cusumsq['cusumsq'], color=COLORS['primary'], linewidth=1.5)
    ax2.plot(t2, cusumsq['upper'], color=COLORS['secondary'],
             linewidth=1, linestyle='--', label='5% bounds')
    ax2.plot(t2, cusumsq['lower'], color=COLORS['secondary'],
             linewidth=1, linestyle='--')
    ax2.fill_between(t2, cusumsq['lower'], cusumsq['upper'],
                     color=COLORS['accent'], alpha=0.08)
    _setup_style(ax2, f'CUSUMSQ Test ({"Stable ✓" if cusumsq["stable"] else "Unstable ✗"})',
                 'Observation', 'CUSUMSQ')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ══════════════════════════════════════════════════════════════════════
# Lag Selection Criteria Plot
# ══════════════════════════════════════════════════════════════════════

def plot_lag_criteria(
    lag_results: pd.DataFrame,
    criterion: str = 'BIC',
    figsize: Tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot information criteria across lag configurations."""
    if lag_results.empty:
        warnings.warn("No lag results to plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    criteria = ['AIC', 'BIC', 'HQ']
    colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]

    x = range(len(lag_results))

    for ic, color in zip(criteria, colors_list):
        if ic in lag_results.columns:
            vals = lag_results[ic].values
            ax.plot(x, vals, color=color, linewidth=1.5, label=ic, alpha=0.8)
            # Mark minimum
            min_idx = np.argmin(vals)
            ax.scatter(min_idx, vals[min_idx], color=color, s=80,
                       zorder=5, marker='*', edgecolors='white', linewidth=0.5)

    _setup_style(ax, 'Information Criteria Across Lag Configurations',
                 'Configuration Index', 'IC Value')
    ax.legend(fontsize=9, framealpha=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ══════════════════════════════════════════════════════════════════════
# Coefficient Comparison Plot
# ══════════════════════════════════════════════════════════════════════

def plot_coefficient_comparison(
    original_results,
    wavelet_results,
    figsize: Tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side coefficient comparison: Original vs Wavelet NARDL.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')

    for ax, results, title in [(ax1, original_results, 'Standard NARDL'),
                                (ax2, wavelet_results, 'Wavelet NARDL')]:
        coef = results.coef_table
        names = coef.index.tolist()
        values = coef['Coefficient'].values
        errors = coef['Std.Error'].values * 1.96
        pvals = coef['p-value'].values

        colors = [COLORS['primary'] if p < 0.05 else COLORS['muted']
                  for p in pvals]

        y_pos = range(len(names))
        ax.barh(y_pos, values, xerr=errors, color=colors,
                alpha=0.7, edgecolor='white', linewidth=0.5,
                capsize=3, error_kw={'linewidth': 0.8})
        ax.axvline(x=0, color=COLORS['muted'], linewidth=0.8, linestyle='-')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        _setup_style(ax, title, 'Coefficient', '')
        ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
