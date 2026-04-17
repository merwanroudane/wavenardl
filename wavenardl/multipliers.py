"""
Dynamic multipliers and bootstrap confidence intervals for NARDL models.

Computes cumulative dynamic multipliers:
    ψ_h⁺ = Σ_{j=0}^{h} ∂y_{t+j} / ∂x_t⁺
    ψ_h⁻ = Σ_{j=0}^{h} ∂y_{t+j} / ∂x_t⁻

The asymmetry between positive and negative multipliers indicates
nonlinear adjustment dynamics.

References
----------
Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .utils import pos_name, neg_name, diff_name, level_lag_name


class DynamicMultipliers:
    """
    Dynamic multiplier computation for NARDL models.

    Parameters
    ----------
    nardl_results : NARDLResults
    horizon : int
        Number of periods (default: 80).
    min_pvalue : float
        Minimum p-value threshold. Coefficients with p > min_pvalue
        are set to zero (default: 0 = include all).
    """

    def __init__(self, nardl_results, horizon: int = 80, min_pvalue: float = 0):
        self.results = nardl_results
        self.horizon = horizon
        self.min_pvalue = min_pvalue

        self.model = nardl_results.model
        self.prep = nardl_results.prep
        self.parsed = self.prep['parsed']
        self.optimal_lags = nardl_results.optimal_lags

        self._compute()

    def _compute(self):
        """Compute dynamic multipliers using omega recursion."""
        model = self.model
        parsed = self.parsed
        h = self.horizon + 1

        param_names = list(model.params.index)
        coeffs = model.params.copy()

        # Zero out insignificant coefficients if requested
        if self.min_pvalue > 0:
            pvals = model.pvalues
            for i, p in enumerate(pvals):
                if p > self.min_pvalue:
                    coeffs.iloc[i] = 0

        dep_var = parsed['dep_var']
        indep_vars = parsed['indep_vars']
        all_asym_lr = parsed['all_asym_lr']
        all_asym_sr = parsed['all_asym_sr']

        # ── Compute omega (autoregressive weights) ───────────────
        dep_lr = level_lag_name(dep_var)
        theta = coeffs.get(dep_lr, 0)  # ρ (speed of adjustment)

        # Get autoregressive lags of dependent
        dep_lag_order = self.optimal_lags.get(dep_var, 1)
        alpha = []
        for lag in range(1, dep_lag_order + 1):
            name = diff_name(dep_var, lag)
            alpha.append(coeffs.get(name, 0))

        p = len(alpha)
        omega = np.zeros(p + 1)
        omega[0] = 1 + theta + (alpha[0] if p > 0 else 0)
        for i in range(1, p):
            omega[i] = alpha[i] - alpha[i - 1]
        if p > 0:
            omega[p] = -alpha[-1]

        self.omega = omega

        # ── Compute lambda and multipliers for each variable ─────
        n_indep = len(indep_vars)
        col_names_pos = []
        col_names_neg = []
        col_names_diff = []

        lambda_mtx = np.zeros((h, n_indep * 2))
        sr_coefs = np.zeros((h, n_indep * 2))

        for v_idx, var in enumerate(indep_vars):
            pn = pos_name(var) if var in all_asym_sr else var
            nn = neg_name(var) if var in all_asym_sr else var

            col_names_pos.append(pn)
            col_names_neg.append(nn)
            col_names_diff.append(f"{var}_diff")

            # Get long-run level coefficients
            if var in all_asym_lr:
                pos_lr = level_lag_name(pos_name(var))
                neg_lr = level_lag_name(neg_name(var))
            else:
                pos_lr = level_lag_name(var)
                neg_lr = pos_lr

            theta_pos = coeffs.get(pos_lr, 0)
            theta_neg = coeffs.get(neg_lr, 0)

            # Get short-run coefficients
            pos_lag = self.optimal_lags.get(pn, self.prep['maxlag'])
            neg_lag = self.optimal_lags.get(nn, self.prep['maxlag'])

            beta_pos = []
            for lag in range(0, pos_lag + 1):
                name = diff_name(pn, lag)
                beta_pos.append(coeffs.get(name, 0))

            beta_neg = []
            for lag in range(0, neg_lag + 1):
                name = diff_name(nn, lag)
                beta_neg.append(coeffs.get(name, 0))

            ci_p = v_idx * 2
            ci_n = v_idx * 2 + 1

            # Fill short-run coefficient matrix
            for i, b in enumerate(beta_pos):
                if i < h:
                    sr_coefs[i, ci_p] = b
            for i, b in enumerate(beta_neg):
                if i < h:
                    sr_coefs[i, ci_n] = b

            # Compute lambda
            lambda_mtx[0, ci_p] = sr_coefs[0, ci_p]
            lambda_mtx[0, ci_n] = sr_coefs[0, ci_n]

            if h > 1:
                lambda_mtx[1, ci_p] = theta_pos - sr_coefs[0, ci_p] + sr_coefs[1, ci_p]
                lambda_mtx[1, ci_n] = theta_neg - sr_coefs[0, ci_n] + sr_coefs[1, ci_n]

            if pn == nn:
                # Symmetric: copy POS to NEG
                lambda_mtx[:, ci_n] = lambda_mtx[:, ci_p]
            else:
                # Higher-order lambda for POS
                for i in range(2, min(len(beta_pos) + 1, h)):
                    lambda_mtx[i, ci_p] = sr_coefs[i, ci_p] - sr_coefs[i - 1, ci_p]
                if len(beta_pos) > 0 and len(beta_pos) + 1 < h:
                    lambda_mtx[len(beta_pos), ci_p] = -sr_coefs[len(beta_pos) - 1, ci_p]

                # Higher-order lambda for NEG
                for i in range(2, min(len(beta_neg) + 1, h)):
                    lambda_mtx[i, ci_n] = sr_coefs[i, ci_n] - sr_coefs[i - 1, ci_n]
                if len(beta_neg) > 0 and len(beta_neg) + 1 < h:
                    lambda_mtx[len(beta_neg), ci_n] = -sr_coefs[len(beta_neg) - 1, ci_n]

        # ── Recursive multiplier computation ─────────────────────
        mpsi_raw = np.zeros((h, n_indep * 2))
        mpsi_raw[0, :] = lambda_mtx[0, :]

        p2 = p + 1
        for t in range(1, h):
            n_say = min(p2, t)
            g_say = max(0, t - p2)
            omega_slice = omega[:n_say]
            mpsi_slice = mpsi_raw[t - 1:g_say - 1 if g_say > 0 else None:-1, :]
            if mpsi_slice.shape[0] < len(omega_slice):
                omega_slice = omega_slice[:mpsi_slice.shape[0]]
            mpsi_raw[t, :] = lambda_mtx[t, :] + omega_slice @ mpsi_slice

        # Cumulative sum
        mpsi_cum = np.cumsum(mpsi_raw, axis=0)

        # ── Build output DataFrame ───────────────────────────────
        columns = []
        data_dict = {'h': np.arange(h)}

        for v_idx, var in enumerate(indep_vars):
            ci_p = v_idx * 2
            ci_n = v_idx * 2 + 1

            pn_label = col_names_pos[v_idx]
            nn_label = col_names_neg[v_idx]
            diff_label = col_names_diff[v_idx]

            data_dict[pn_label] = mpsi_cum[:, ci_p]
            data_dict[nn_label] = -mpsi_cum[:, ci_n]  # Reverse sign for display
            data_dict[diff_label] = mpsi_cum[:, ci_p] - mpsi_cum[:, ci_n]

        self.mpsi = pd.DataFrame(data_dict)
        self.lambda_mtx = lambda_mtx
        self.sr_coefs = sr_coefs
        self.indep_vars = indep_vars

    def get_multipliers(self, variable: Optional[str] = None) -> pd.DataFrame:
        """
        Get multiplier DataFrame.

        Parameters
        ----------
        variable : str or None
            If specified, return only multipliers for this variable.
        """
        if variable is None:
            return self.mpsi

        cols = ['h']
        for col in self.mpsi.columns:
            if variable in col:
                cols.append(col)

        if len(cols) == 1:
            raise ValueError(f"Variable '{variable}' not found in multipliers.")
        return self.mpsi[cols]

    def summary(self, print_output: bool = True) -> str:
        """Print formatted multiplier summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("          DYNAMIC MULTIPLIERS")
        lines.append("=" * 70)
        lines.append(f"  Horizon         : {self.horizon}")
        lines.append(f"  Min p-value     : {self.min_pvalue}")
        lines.append("-" * 70)

        # Show initial and final values for each variable
        for var in self.indep_vars:
            pn = pos_name(var) if var in self.parsed['all_asym_sr'] else var
            nn = neg_name(var) if var in self.parsed['all_asym_sr'] else var

            if pn in self.mpsi.columns:
                lines.append(f"\n  Variable: {var}")
                lines.append(f"    {'h':>5s}  {'Positive':>12s}  {'Negative':>12s}  "
                             f"{'Asymmetry':>12s}")
                lines.append(f"    {'─' * 50}")

                diff_col = f"{var}_diff"
                for t in [0, 5, 10, 20, 40, self.horizon]:
                    if t < len(self.mpsi):
                        row = self.mpsi.iloc[t]
                        p_val = row.get(pn, 0)
                        n_val = row.get(nn, 0)
                        d_val = row.get(diff_col, 0)
                        lines.append(f"    {t:>5d}  {p_val:>12.4f}  {n_val:>12.4f}  "
                                     f"{d_val:>12.4f}")

        lines.append("-" * 70)
        lines.append("=" * 70)

        text = "\n".join(lines)
        if print_output:
            print(text)
        return text


def bootstrap_multipliers(
    nardl_results,
    horizon: int = 80,
    replications: int = 100,
    confidence_level: float = 95,
    min_pvalue: float = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for dynamic multiplier asymmetry.

    Parameters
    ----------
    nardl_results : NARDLResults
    horizon : int
    replications : int
    confidence_level : float (e.g. 95 for 95% CI)
    min_pvalue : float
    verbose : bool

    Returns
    -------
    pd.DataFrame with multipliers and CI columns
    """
    # Base multipliers
    base_mp = DynamicMultipliers(nardl_results, horizon, min_pvalue)
    mpsi = base_mp.mpsi.copy()

    parsed = base_mp.parsed
    sr_asym = parsed['all_asym_sr']

    if not sr_asym:
        if verbose:
            print("No asymmetric variables — bootstrap CI not needed.")
        return mpsi

    # Collect bootstrap samples
    model = nardl_results.model
    resid = model.resid.values
    n = len(resid)

    diff_columns = {var: f"{var}_diff" for var in base_mp.indep_vars}
    boot_diffs = {var: [] for var in base_mp.indep_vars}

    for r in range(replications):
        if verbose and (r + 1) % 20 == 0:
            print(f"  Bootstrap replication {r + 1}/{replications}...")

        # Resample residuals
        boot_resid = resid[np.random.choice(n, size=n, replace=True)]

        # Generate new dependent variable
        y_fitted = model.fittedvalues.values
        y_boot = y_fitted + boot_resid[:len(y_fitted)]

        # Re-estimate model with bootstrapped y
        try:
            X = model.model.exog
            import statsmodels.api as sm
            boot_model = sm.OLS(y_boot, X).fit()

            # Create a mock results object
            from types import SimpleNamespace
            mock_results = SimpleNamespace()
            mock_results.model = boot_model
            mock_results.prep = nardl_results.prep
            mock_results.optimal_lags = nardl_results.optimal_lags
            mock_results.parsed = parsed

            # Compute multipliers
            boot_mp = DynamicMultipliers.__new__(DynamicMultipliers)
            boot_mp.results = mock_results
            boot_mp.horizon = horizon
            boot_mp.min_pvalue = min_pvalue
            boot_mp.model = boot_model
            boot_mp.prep = nardl_results.prep
            boot_mp.parsed = parsed
            boot_mp.optimal_lags = nardl_results.optimal_lags
            boot_mp._compute()

            for var in base_mp.indep_vars:
                diff_col = f"{var}_diff"
                if diff_col in boot_mp.mpsi.columns:
                    boot_diffs[var].append(boot_mp.mpsi[diff_col].values)
        except Exception:
            continue

    # Compute confidence intervals
    alpha = (100 - confidence_level) / 100
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    for var in base_mp.indep_vars:
        if boot_diffs[var]:
            boot_array = np.array(boot_diffs[var])
            lower_ci = np.percentile(boot_array, lower_q * 100, axis=0)
            upper_ci = np.percentile(boot_array, upper_q * 100, axis=0)
            mpsi[f"{var}_lCI"] = lower_ci
            mpsi[f"{var}_uCI"] = upper_ci

    return mpsi
