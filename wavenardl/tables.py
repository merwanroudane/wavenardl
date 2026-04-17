"""
Publication-quality tables for NARDL model results.

Supports console (rich), LaTeX, and HTML output formats.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .utils import significance_stars


def summary_table(nardl_results, format: str = 'console') -> str:
    """
    Generate publication-quality summary table.

    Parameters
    ----------
    nardl_results : NARDLResults
    format : str, 'console', 'latex', or 'html'

    Returns
    -------
    str : formatted table
    """
    if format == 'latex':
        return _summary_latex(nardl_results)
    elif format == 'html':
        return _summary_html(nardl_results)
    else:
        return nardl_results.summary(print_output=False)


def comparison_table(
    original_results,
    wavelet_results,
    format: str = 'console',
) -> str:
    """
    Generate side-by-side comparison table.
    """
    o = original_results
    w = wavelet_results

    # Merge coefficient tables
    o_coef = o.coef_table[['Coefficient', 'Std.Error', 'Signif']].copy()
    w_coef = w.coef_table[['Coefficient', 'Std.Error', 'Signif']].copy()

    o_coef.columns = ['Orig_Coef', 'Orig_SE', 'Orig_Sig']
    w_coef.columns = ['Wave_Coef', 'Wave_SE', 'Wave_Sig']

    merged = o_coef.join(w_coef, how='outer')

    if format == 'latex':
        return _comparison_latex(merged, o, w)
    elif format == 'html':
        return _comparison_html(merged, o, w)
    else:
        return _comparison_console(merged, o, w)


def bounds_test_table(test_result, format: str = 'console') -> str:
    """Format PSS bounds test results as a table."""
    lines = []
    lines.append("=" * 65)
    lines.append("  PSS BOUNDS TEST FOR COINTEGRATION")
    lines.append("=" * 65)
    lines.append(f"  F-statistic : {test_result.statistic:.4f}")
    lines.append(f"  Case        : {test_result.details.get('Case', '')}")
    lines.append(f"  k           : {test_result.details.get('k (regressors)', '')}")
    lines.append(f"  Decision    : {test_result.decision}")
    lines.append("-" * 65)

    if 'Critical Values' in test_result.details:
        cv = test_result.details['Critical Values']
        lines.append(f"  {'Level':>10s}  {'I(0)':>10s}  {'I(1)':>10s}")
        lines.append(f"  {'─'*35}")
        for idx, row in cv.iterrows():
            lines.append(f"  {idx:>10s}  {row['I(0)']:>10.3f}  {row['I(1)']:>10.3f}")

    lines.append("=" * 65)
    return "\n".join(lines)


def diagnostics_table(diag_df: pd.DataFrame, format: str = 'console') -> str:
    """Format diagnostics results as a table."""
    if format == 'latex':
        return _diagnostics_latex(diag_df)
    else:
        return diag_df.to_string(float_format='{:.4f}'.format)


# ── Internal formatters ──────────────────────────────────────────────

def _summary_latex(results) -> str:
    """Generate LaTeX table."""
    ct = results.coef_table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{NARDL Estimation Results}",
        r"\label{tab:nardl_results}",
        r"\begin{tabular}{lcccc}",
        r"\hline\hline",
        r"Variable & Coefficient & Std. Error & t-value & \\",
        r"\hline",
    ]

    for idx, row in ct.iterrows():
        name = idx.replace('_', r'\_')
        stars = row['Signif']
        lines.append(
            f"  {name} & {row['Coefficient']:.4f}{stars} & "
            f"({row['Std.Error']:.4f}) & {row['t-value']:.3f} \\\\"
        )

    lines.extend([
        r"\hline",
        f"Observations & \\multicolumn{{4}}{{c}}{{{results.nobs}}} \\\\",
        f"R$^2$ & \\multicolumn{{4}}{{c}}{{{results.rsquared:.4f}}} \\\\",
        f"Adj. R$^2$ & \\multicolumn{{4}}{{c}}{{{results.rsquared_adj:.4f}}} \\\\",
        f"AIC & \\multicolumn{{4}}{{c}}{{{results.aic:.2f}}} \\\\",
        f"BIC & \\multicolumn{{4}}{{c}}{{{results.bic:.2f}}} \\\\",
        r"\hline\hline",
        r"\multicolumn{5}{l}{\footnotesize $^{***}p<0.01$; $^{**}p<0.05$; $^{*}p<0.10$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _summary_html(results) -> str:
    """Generate styled HTML table."""
    ct = results.coef_table
    style = """
    <style>
    .nardl-table { border-collapse: collapse; font-family: 'Inter', sans-serif;
                   font-size: 13px; margin: 20px 0; }
    .nardl-table th { background: linear-gradient(135deg, #667eea, #764ba2);
                      color: white; padding: 10px 15px; text-align: left; }
    .nardl-table td { padding: 8px 15px; border-bottom: 1px solid #e5e7eb; }
    .nardl-table tr:hover { background: #f8fafc; }
    .sig { color: #dc2626; font-weight: bold; }
    .footer { font-size: 11px; color: #9ca3af; margin-top: 5px; }
    </style>
    """

    html = style + '<table class="nardl-table">\n'
    html += '<tr><th>Variable</th><th>Coefficient</th><th>Std.Error</th>'
    html += '<th>t-value</th><th>p-value</th></tr>\n'

    for idx, row in ct.iterrows():
        sig_class = ' class="sig"' if row['p-value'] < 0.05 else ''
        html += f'<tr><td{sig_class}>{idx}</td>'
        html += f'<td>{row["Coefficient"]:.4f}{row["Signif"]}</td>'
        html += f'<td>{row["Std.Error"]:.4f}</td>'
        html += f'<td>{row["t-value"]:.3f}</td>'
        html += f'<td>{row["p-value"]:.4f}</td></tr>\n'

    html += '</table>\n'
    html += f'<p class="footer">N={results.nobs}, R²={results.rsquared:.4f}, '
    html += f'Adj.R²={results.rsquared_adj:.4f}, AIC={results.aic:.2f}, '
    html += f'BIC={results.bic:.2f}</p>\n'
    html += '<p class="footer">*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.10</p>'

    return html


def _comparison_console(merged, orig, wave) -> str:
    """Console comparison table."""
    lines = []
    lines.append("=" * 78)
    lines.append("  COEFFICIENT COMPARISON: Original vs Wavelet NARDL")
    lines.append("=" * 78)
    lines.append(f"  {'Variable':<22s} {'Original':>12s} {'':>5s} "
                 f"{'Wavelet':>12s} {'':>5s}")
    lines.append("-" * 78)

    for idx, row in merged.iterrows():
        o_val = f"{row['Orig_Coef']:.4f}" if not np.isnan(row.get('Orig_Coef', np.nan)) else '—'
        o_se = f"({row['Orig_SE']:.4f})" if not np.isnan(row.get('Orig_SE', np.nan)) else ''
        o_sig = str(row.get('Orig_Sig', ''))
        w_val = f"{row['Wave_Coef']:.4f}" if not np.isnan(row.get('Wave_Coef', np.nan)) else '—'
        w_se = f"({row['Wave_SE']:.4f})" if not np.isnan(row.get('Wave_SE', np.nan)) else ''
        w_sig = str(row.get('Wave_Sig', ''))

        lines.append(f"  {str(idx):<22s} {o_val:>12s} {o_sig:>5s} "
                     f"{w_val:>12s} {w_sig:>5s}")
        lines.append(f"  {'':22s} {o_se:>12s} {'':5s} {w_se:>12s}")

    lines.append("-" * 78)
    lines.append(f"  {'AIC':<22s} {orig.aic:>12.2f} {'':>5s} {wave.aic:>12.2f}")
    lines.append(f"  {'BIC':<22s} {orig.bic:>12.2f} {'':>5s} {wave.bic:>12.2f}")
    lines.append(f"  {'R²':<22s} {orig.rsquared:>12.4f} {'':>5s} {wave.rsquared:>12.4f}")
    lines.append("=" * 78)

    return "\n".join(lines)


def _comparison_latex(merged, orig, wave) -> str:
    """LaTeX comparison table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison: Standard NARDL vs Wavelet NARDL}",
        r"\begin{tabular}{lcc}",
        r"\hline\hline",
        r"Variable & Standard NARDL & Wavelet NARDL \\",
        r"\hline",
    ]

    for idx, row in merged.iterrows():
        name = str(idx).replace('_', r'\_')
        o_val = f"{row['Orig_Coef']:.4f}{row.get('Orig_Sig', '')}" \
            if not np.isnan(row.get('Orig_Coef', np.nan)) else '—'
        w_val = f"{row['Wave_Coef']:.4f}{row.get('Wave_Sig', '')}" \
            if not np.isnan(row.get('Wave_Coef', np.nan)) else '—'
        o_se = f"({row['Orig_SE']:.4f})" if not np.isnan(row.get('Orig_SE', np.nan)) else ''
        w_se = f"({row['Wave_SE']:.4f})" if not np.isnan(row.get('Wave_SE', np.nan)) else ''

        lines.append(f"  {name} & {o_val} & {w_val} \\\\")
        lines.append(f"  & {o_se} & {w_se} \\\\")

    lines.extend([
        r"\hline",
        f"  AIC & {orig.aic:.2f} & {wave.aic:.2f} \\\\",
        f"  BIC & {orig.bic:.2f} & {wave.bic:.2f} \\\\",
        f"  R$^2$ & {orig.rsquared:.4f} & {wave.rsquared:.4f} \\\\",
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _comparison_html(merged, orig, wave) -> str:
    """HTML comparison table."""
    html = _summary_html.__doc__ or ''  # placeholder
    return merged.to_html(float_format='{:.4f}'.format)


def _diagnostics_latex(diag_df) -> str:
    """LaTeX diagnostics table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Model Diagnostics}",
        r"\begin{tabular}{lccc}",
        r"\hline\hline",
        r"Test & Statistic & p-value & Decision \\",
        r"\hline",
    ]

    for idx, row in diag_df.iterrows():
        name = str(idx).replace('_', r'\_')
        stat = f"{row['Statistic']:.4f}" if not np.isnan(row['Statistic']) else '—'
        pval = f"{row['p-value']:.4f}" if not np.isnan(row['p-value']) else '—'
        lines.append(f"  {name} & {stat} & {pval} & {row['Decision']} \\\\")

    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)
