"""
Long-run multipliers via delta method for NARDL models.

Computes long-run coefficients: LR_i = -θ_i / ρ
with standard errors via the delta method.

References
----------
Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014).
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Dict, Optional

from .utils import level_lag_name, pos_name, neg_name


def compute_long_run_multipliers(nardl_results) -> pd.DataFrame:
    """
    Calculate long-run multipliers from NARDL results.

    LR_i = -θ_i / ρ

    where ρ is the coefficient of y_{t-1} and θ_i are coefficients
    of x_i_{t-1} (or x_i_POS_{t-1}, x_i_NEG_{t-1}).

    Standard errors are computed via delta method:
        Var(LR_i) = A² Var(θ_i) + 2AB Cov(θ_i, ρ) + B² Var(ρ)
    where A = -1/ρ and B = θ_i/ρ²

    Parameters
    ----------
    nardl_results : NARDLResults

    Returns
    -------
    pd.DataFrame with columns:
        LR_Coefficient, Std.Error, t-value, p-value, Signif
    """
    model = nardl_results.model
    prep = nardl_results.prep
    parsed = prep['parsed']

    dep_var = parsed['dep_var']
    indep_vars = parsed['indep_vars']
    lr_asym = parsed['all_asym_lr']

    param_names = list(model.params.index)
    coeffs = model.params
    vcov = model.cov_params()

    # Dependent variable level lag coefficient (ρ)
    dep_lr = level_lag_name(dep_var)
    if dep_lr not in param_names:
        raise ValueError(f"Dependent level lag '{dep_lr}' not found in model.")
    rho = coeffs[dep_lr]
    dep_idx = param_names.index(dep_lr)

    if abs(rho) < 1e-12:
        raise ValueError("Speed of adjustment (ρ) is essentially zero. "
                         "Cannot compute long-run multipliers.")

    results = {}

    # Intercept long-run if present
    if 'const' in param_names:
        const_val = coeffs['const']
        lr_const = -const_val / rho
        const_idx = param_names.index('const')

        A = -1.0 / rho
        B = const_val / (rho ** 2)
        var_lr = (A ** 2 * vcov.iloc[const_idx, const_idx] +
                  2 * A * B * vcov.iloc[const_idx, dep_idx] +
                  B ** 2 * vcov.iloc[dep_idx, dep_idx])
        se_lr = np.sqrt(max(var_lr, 0))

        results['Intercept'] = {
            'LR_Coefficient': lr_const,
            'Std.Error': se_lr,
        }

    # Independent variables
    for var in indep_vars:
        if var in lr_asym:
            # Asymmetric: compute LR for POS and NEG separately
            for suffix_fn, suffix_label in [(pos_name, 'POS'), (neg_name, 'NEG')]:
                asym_var = suffix_fn(var)
                lr_name = level_lag_name(asym_var)

                if lr_name in param_names:
                    theta_i = coeffs[lr_name]
                    theta_idx = param_names.index(lr_name)

                    lr_i = -theta_i / rho

                    A = -1.0 / rho
                    B = theta_i / (rho ** 2)
                    var_lr = (A ** 2 * vcov.iloc[theta_idx, theta_idx] +
                              2 * A * B * vcov.iloc[theta_idx, dep_idx] +
                              B ** 2 * vcov.iloc[dep_idx, dep_idx])
                    se_lr = np.sqrt(max(var_lr, 0))

                    label = f"L({var}_{suffix_label})"
                    results[label] = {
                        'LR_Coefficient': lr_i,
                        'Std.Error': se_lr,
                    }
        else:
            # Symmetric
            lr_name = level_lag_name(var)
            if lr_name in param_names:
                theta_i = coeffs[lr_name]
                theta_idx = param_names.index(lr_name)

                lr_i = -theta_i / rho

                A = -1.0 / rho
                B = theta_i / (rho ** 2)
                var_lr = (A ** 2 * vcov.iloc[theta_idx, theta_idx] +
                          2 * A * B * vcov.iloc[theta_idx, dep_idx] +
                          B ** 2 * vcov.iloc[dep_idx, dep_idx])
                se_lr = np.sqrt(max(var_lr, 0))

                results[f"L({var})"] = {
                    'LR_Coefficient': lr_i,
                    'Std.Error': se_lr,
                }

    # Build DataFrame
    df = pd.DataFrame(results).T
    df.index.name = 'Variable'
    nobs = model.nobs
    k = len(model.params)
    df_resid = nobs - k

    df['t-value'] = df['LR_Coefficient'] / df['Std.Error']
    df['p-value'] = 2 * scipy_stats.t.sf(np.abs(df['t-value']), df_resid)

    from .utils import significance_stars
    df['Signif'] = df['p-value'].apply(significance_stars)

    return df


def long_run_summary(nardl_results, print_output: bool = True) -> str:
    """Print formatted long-run multipliers table."""
    lr_df = compute_long_run_multipliers(nardl_results)

    lines = []
    lines.append("=" * 70)
    lines.append("          LONG-RUN MULTIPLIERS (Delta Method)")
    lines.append("=" * 70)
    lines.append(f"  {'Variable':<20s} {'LR Coef':>10s} {'Std.Err':>10s} "
                 f"{'t-value':>10s} {'p-value':>10s} {'':>5s}")
    lines.append("-" * 70)

    for idx, row in lr_df.iterrows():
        lines.append(
            f"  {str(idx):<20s} {row['LR_Coefficient']:>10.4f} {row['Std.Error']:>10.4f} "
            f"{row['t-value']:>10.4f} {row['p-value']:>10.4f} {row['Signif']:>5s}"
        )

    lines.append("-" * 70)
    lines.append("  LR_i = -θ_i / ρ ;  SE via delta method")
    lines.append("  Signif.: '***' 0.01, '**' 0.05, '*' 0.10")
    lines.append("=" * 70)

    text = "\n".join(lines)
    if print_output:
        print(text)
    return text
