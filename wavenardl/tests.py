"""
Statistical tests for NARDL models:
- PSS Bounds F-test (Pesaran et al., 2001)
- PSS t-test for ECM
- Narayan small-sample F-test (Narayan, 2005)
- Symmetry tests (Shin et al., 2014)

References
----------
Pesaran, M. H., Shin, Y., & Smith, R. (2001).
Narayan, P. K. (2005).
Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014).
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Dict, Optional, Tuple, List

from .critical_values import get_all_pss_critical_values, pss_decision
from .utils import level_lag_name, pos_name, neg_name, diff_name


class TestResult:
    """Container for test results with pretty printing."""

    def __init__(self, test_name, statistic, p_value=None, decision=None,
                 details=None, hypothesis=None):
        self.test_name = test_name
        self.statistic = statistic
        self.p_value = p_value
        self.decision = decision
        self.details = details or {}
        self.hypothesis = hypothesis or {}

    def __repr__(self):
        lines = [f"\n{'='*60}",
                 f"  {self.test_name}",
                 f"{'='*60}"]

        if isinstance(self.statistic, dict):
            for k, v in self.statistic.items():
                lines.append(f"  {k:<20s}: {v:.4f}")
        else:
            lines.append(f"  Test Statistic    : {self.statistic:.4f}")

        if self.p_value is not None:
            lines.append(f"  P-value           : {self.p_value:.6f}")

        if self.decision:
            lines.append(f"  Decision          : {self.decision}")

        if self.details:
            lines.append(f"{'─'*60}")
            for k, v in self.details.items():
                if isinstance(v, pd.DataFrame):
                    lines.append(f"  {k}:")
                    lines.append(v.to_string(index=True, float_format='{:.4f}'.format))
                elif isinstance(v, float):
                    lines.append(f"  {k:<20s}: {v:.4f}")
                else:
                    lines.append(f"  {k:<20s}: {v}")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# PSS Bounds F-Test
# ══════════════════════════════════════════════════════════════════════

def pss_f_test(nardl_results, case: int = 3, sig: str = 'auto') -> TestResult:
    """
    Pesaran-Shin-Smith (PSS) Bounds F-test for Cointegration.

    Tests joint significance of lagged level variables:
        H₀: ρ = θ₁ = θ₂ = ... = θ_k = 0  (no cointegration)
        H₁: At least one ≠ 0               (cointegration)

    Parameters
    ----------
    nardl_results : NARDLResults
    case : int, 1-5
    sig : str, significance level or 'auto'

    Returns
    -------
    TestResult
    """
    model = nardl_results.model
    prep = nardl_results.prep
    parsed = prep['parsed']

    # Auto-detect case
    if parsed['no_constant']:
        case = 1
    elif parsed['trend'] and case < 4:
        case = 5

    # Get long-run variable names
    lr_names = prep['long_run_vars']

    # Add restricted terms based on case
    restricted_names = list(lr_names)
    if case == 2:
        restricted_names.append('const')
    elif case == 4:
        restricted_names.append('trend')

    # Build restriction matrix for F-test
    param_names = list(model.params.index)
    n_restrictions = 0
    R_rows = []

    for rname in restricted_names:
        if rname in param_names:
            row = np.zeros(len(param_names))
            idx = param_names.index(rname)
            row[idx] = 1.0
            R_rows.append(row)
            n_restrictions += 1

    if n_restrictions == 0:
        raise ValueError("No long-run variables found for bounds test.")

    R = np.array(R_rows)
    r = np.zeros(n_restrictions)

    # Compute F-statistic
    f_result = model.f_test(R)
    F_stat = float(f_result.fvalue)

    # Number of independent long-run regressors (excluding dependent)
    k = len(lr_names) - 1  # subtract the dependent variable's level lag

    # Get critical values and decision
    result = pss_decision(F_stat, case, k, sig)

    # Build critical values table
    all_cvs = get_all_pss_critical_values(case, min(k, 10))
    cv_df = pd.DataFrame(
        {sig_level: {'I(0)': bounds[0], 'I(1)': bounds[1]}
         for sig_level, bounds in all_cvs.items()}
    ).T
    cv_df.index.name = 'Significance'

    case_names = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V'}

    return TestResult(
        test_name='Pesaran-Shin-Smith (PSS) Bounds F-test',
        statistic=F_stat,
        p_value=None,
        decision=result['decision'],
        details={
            'Case': f"{case_names.get(case, case)}",
            'k (regressors)': k,
            'Observations': int(model.nobs),
            'Critical Values': cv_df,
        },
        hypothesis={
            'H0': 'No cointegration (level variables jointly insignificant)',
            'H1': 'Cointegration exists',
        }
    )


# ══════════════════════════════════════════════════════════════════════
# Symmetry Tests
# ══════════════════════════════════════════════════════════════════════

def symmetry_test(nardl_results) -> TestResult:
    """
    Wald tests for short-run and long-run symmetry.

    Long-run symmetry:
        H₀: -θ⁺/ρ = -θ⁻/ρ  (symmetric long-run multipliers)
    Short-run symmetry:
        H₀: Σβ⁺_j = Σβ⁻_j  (symmetric short-run dynamics)

    Parameters
    ----------
    nardl_results : NARDLResults

    Returns
    -------
    TestResult with long-run and short-run Wald test results.
    """
    model = nardl_results.model
    prep = nardl_results.prep
    parsed = prep['parsed']
    param_names = list(model.params.index)
    optimal_lags = nardl_results.optimal_lags

    lr_asym = parsed['all_asym_lr']
    sr_asym = parsed['all_asym_sr']

    results_lr = {}
    results_sr = {}

    # ── Long-run symmetry tests ──────────────────────────────────
    dep_var = parsed['dep_var']
    dep_lr_name = level_lag_name(dep_var)

    if dep_lr_name not in param_names:
        pass  # Skip if not found
    else:
        dep_lr_idx = param_names.index(dep_lr_name)

        for var in lr_asym:
            pn = pos_name(var)
            nn = neg_name(var)
            pos_lr = level_lag_name(pn)
            neg_lr = level_lag_name(nn)

            if pos_lr in param_names and neg_lr in param_names:
                pos_idx = param_names.index(pos_lr)
                neg_idx = param_names.index(neg_lr)

                # Test: -coef_pos/coef_dep = -coef_neg/coef_dep
                # Equivalent to: coef_pos = coef_neg (when dep != 0)
                # Use linearHypothesis equivalent
                R = np.zeros(len(param_names))
                R[pos_idx] = 1.0
                R[neg_idx] = -1.0

                try:
                    f_result = model.f_test(R.reshape(1, -1))
                    F_val = float(f_result.fvalue)
                    p_val = float(f_result.pvalue)

                    results_lr[var] = {
                        'F-statistic': F_val,
                        'p-value': p_val,
                        'Decision': 'Asymmetric' if p_val < 0.05 else 'Symmetric',
                        'H0': f'L({pn}) = L({nn})',
                    }
                except Exception:
                    pass

    # ── Short-run symmetry tests ─────────────────────────────────
    for var in sr_asym:
        pn = pos_name(var)
        nn = neg_name(var)

        # Collect all short-run coefficients for POS and NEG
        pos_names = []
        neg_names = []
        max_var_lag = max(
            optimal_lags.get(pn, nardl_results.prep['maxlag']),
            optimal_lags.get(nn, nardl_results.prep['maxlag']),
        )

        for lag in range(0, max_var_lag + 1):
            p_name = diff_name(pn, lag)
            n_name = diff_name(nn, lag)
            if p_name in param_names:
                pos_names.append(p_name)
            if n_name in param_names:
                neg_names.append(n_name)

        if pos_names and neg_names:
            # Test: Σ coef_pos = Σ coef_neg
            R = np.zeros(len(param_names))
            for pname in pos_names:
                R[param_names.index(pname)] = 1.0
            for nname in neg_names:
                R[param_names.index(nname)] = -1.0

            try:
                f_result = model.f_test(R.reshape(1, -1))
                F_val = float(f_result.fvalue)
                p_val = float(f_result.pvalue)

                results_sr[var] = {
                    'F-statistic': F_val,
                    'p-value': p_val,
                    'Decision': 'Asymmetric' if p_val < 0.05 else 'Symmetric',
                    'H0': f'Σβ⁺({var}) = Σβ⁻({var})',
                }
            except Exception:
                pass

    # Build summary tables
    details = {}
    if results_lr:
        lr_df = pd.DataFrame(results_lr).T
        lr_df.index.name = 'Variable'
        details['Long-run Symmetry'] = lr_df

    if results_sr:
        sr_df = pd.DataFrame(results_sr).T
        sr_df.index.name = 'Variable'
        details['Short-run Symmetry'] = sr_df

    return TestResult(
        test_name='Wald Symmetry Tests',
        statistic={'n_LR_tests': len(results_lr), 'n_SR_tests': len(results_sr)},
        decision=None,
        details=details,
        hypothesis={
            'H0 (LR)': 'Long-run positive and negative effects are equal',
            'H0 (SR)': 'Short-run positive and negative effects are equal',
        }
    )
