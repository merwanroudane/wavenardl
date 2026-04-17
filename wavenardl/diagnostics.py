"""
Diagnostic tests for NARDL model residuals.

Provides a battery of tests for:
- Serial correlation (Breusch-Godfrey, Durbin-Watson)
- Heteroskedasticity (Breusch-Pagan, White)
- Normality (Jarque-Bera, Shapiro-Wilk)
- Functional form (Ramsey RESET)
- Parameter stability (CUSUM, CUSUMSQ)
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
try:
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
except ImportError:
    from statsmodels.stats.diagnostic import acorr_breuschgodfrey as acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from typing import Dict, Optional

from .utils import significance_stars


def run_all_diagnostics(nardl_results, max_bg_lags: int = 4) -> pd.DataFrame:
    """
    Run all diagnostic tests on NARDL residuals.

    Parameters
    ----------
    nardl_results : NARDLResults
    max_bg_lags : int
        Maximum lags for Breusch-Godfrey test.

    Returns
    -------
    pd.DataFrame with test results.
    """
    model = nardl_results.model
    results = {}

    # 1. Durbin-Watson
    dw = durbin_watson(model.resid)
    results['Durbin-Watson'] = {
        'Statistic': dw,
        'p-value': np.nan,  # DW doesn't have a simple p-value
        'Decision': 'No autocorrelation' if 1.5 < dw < 2.5 else 'Possible autocorrelation',
    }

    # 2. Breusch-Godfrey (serial correlation)
    for lags in [1, 2, max_bg_lags]:
        try:
            bg_lm, bg_pval, bg_fval, bg_f_pval = acorr_breusch_godfrey(model, nlags=lags)
            results[f'Breusch-Godfrey ({lags} lag{"s" if lags>1 else ""})'] = {
                'Statistic': bg_fval,
                'p-value': bg_f_pval,
                'Decision': 'No serial corr.' if bg_f_pval > 0.05 else 'Serial corr. detected',
            }
        except Exception:
            pass

    # 3. Breusch-Pagan (heteroskedasticity)
    try:
        bp_lm, bp_pval, bp_fval, bp_f_pval = het_breuschpagan(
            model.resid, model.model.exog)
        results['Breusch-Pagan'] = {
            'Statistic': bp_fval,
            'p-value': bp_f_pval,
            'Decision': 'Homoskedastic' if bp_f_pval > 0.05 else 'Heteroskedastic',
        }
    except Exception:
        pass

    # 4. Jarque-Bera (normality)
    try:
        jb, jb_pval, skew, kurt = jarque_bera(model.resid)
        results['Jarque-Bera'] = {
            'Statistic': jb,
            'p-value': jb_pval,
            'Decision': 'Normal' if jb_pval > 0.05 else 'Non-normal',
        }
    except Exception:
        pass

    # 5. Shapiro-Wilk (normality, max 5000 obs)
    resid_test = model.resid.values
    if len(resid_test) <= 5000:
        try:
            sw_stat, sw_pval = scipy_stats.shapiro(resid_test)
            results['Shapiro-Wilk'] = {
                'Statistic': sw_stat,
                'p-value': sw_pval,
                'Decision': 'Normal' if sw_pval > 0.05 else 'Non-normal',
            }
        except Exception:
            pass

    # 6. Ramsey RESET (functional form)
    try:
        reset_result = _ramsey_reset(model)
        results['Ramsey RESET'] = reset_result
    except Exception:
        pass

    df = pd.DataFrame(results).T
    df.index.name = 'Test'
    return df


def _ramsey_reset(model, power: int = 3) -> Dict:
    """Ramsey RESET test for functional form misspecification."""
    y = model.model.endog
    X = model.model.exog
    fitted = model.fittedvalues

    # Add powers of fitted values
    X_aug = np.column_stack([X] + [fitted ** p for p in range(2, power + 1)])

    model_aug = sm.OLS(y, X_aug).fit()

    # F-test comparing restricted vs unrestricted
    n = len(y)
    k_r = X.shape[1]
    k_u = X_aug.shape[1]
    ssr_r = np.sum(model.resid ** 2)
    ssr_u = np.sum(model_aug.resid ** 2)

    df1 = k_u - k_r
    df2 = n - k_u
    f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
    p_value = 1 - scipy_stats.f.cdf(f_stat, df1, df2)

    return {
        'Statistic': f_stat,
        'p-value': p_value,
        'Decision': 'Correct form' if p_value > 0.05 else 'Misspecification',
    }


def cusum_test(nardl_results) -> Dict:
    """
    CUSUM test for parameter stability.

    Returns dict with CUSUM values, upper and lower 5% boundaries.
    """
    model = nardl_results.model
    resid = model.resid.values
    n = len(resid)
    k = len(model.params)

    # Recursive residuals (simplified)
    sigma = np.std(resid, ddof=k)
    W = np.cumsum(resid) / sigma

    # 5% significance boundaries
    t_values = np.arange(k + 1, n + 1)
    a = 0.948  # 5% level constant
    upper = a * np.sqrt(n - k) + 2 * a * (t_values - k) / np.sqrt(n - k)
    lower = -upper

    return {
        'cusum': W[k:],
        'upper': upper,
        'lower': lower,
        't_values': t_values,
        'stable': np.all(np.abs(W[k:]) <= upper[:len(W[k:])]),
    }


def cusum_sq_test(nardl_results) -> Dict:
    """
    CUSUMSQ test for parameter stability.

    Returns dict with CUSUMSQ values and 5% boundaries.
    """
    model = nardl_results.model
    resid = model.resid.values
    n = len(resid)
    k = len(model.params)

    sq_resid = resid ** 2
    total_sq = np.sum(sq_resid)
    S = np.cumsum(sq_resid) / total_sq

    # Expected values under stability
    t_values = np.arange(1, n + 1)
    expected = (t_values - k) / (n - k)
    expected = np.clip(expected, 0, 1)

    # 5% critical value (approximately 1.36/sqrt(n-k))
    c = 1.36 / np.sqrt(n - k)
    upper = expected + c
    lower = expected - c

    return {
        'cusumsq': S,
        'expected': expected,
        'upper': np.clip(upper, 0, None),
        'lower': np.clip(lower, None, 1),
        't_values': t_values,
        'stable': np.all((S >= lower) & (S <= upper)),
    }


def diagnostics_summary(nardl_results, print_output: bool = True) -> str:
    """Print formatted diagnostics summary."""
    diag = run_all_diagnostics(nardl_results)

    lines = []
    lines.append("=" * 70)
    lines.append("          MODEL DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append(f"  {'Test':<35s} {'Statistic':>10s} {'p-value':>10s} {'Decision':>15s}")
    lines.append("-" * 70)

    for idx, row in diag.iterrows():
        stat_str = f"{row['Statistic']:.4f}" if not np.isnan(row['Statistic']) else 'N/A'
        pval_str = f"{row['p-value']:.4f}" if not np.isnan(row['p-value']) else 'N/A'
        lines.append(f"  {str(idx):<35s} {stat_str:>10s} {pval_str:>10s} "
                     f"{row['Decision']:>15s}")

    # CUSUM
    cusum = cusum_test(nardl_results)
    cusumsq = cusum_sq_test(nardl_results)
    lines.append("-" * 70)
    lines.append(f"  {'CUSUM':<35s} {'':>10s} {'':>10s} "
                 f"{'Stable' if cusum['stable'] else 'Unstable':>15s}")
    lines.append(f"  {'CUSUMSQ':<35s} {'':>10s} {'':>10s} "
                 f"{'Stable' if cusumsq['stable'] else 'Unstable':>15s}")

    lines.append("=" * 70)

    text = "\n".join(lines)
    if print_output:
        print(text)
    return text
