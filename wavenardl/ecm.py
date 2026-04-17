"""
Error Correction Model (ECM) representation for NARDL models.

Implements the two-step ECM approach following Pesaran et al. (2001)
and Shin et al. (2014), matching the R kardl ecm() function.

Step 1: Estimate long-run equation → extract residuals
Step 2: Include lagged residuals as error correction term in short-run equation

Cases 1-5 for deterministic specification.

References
----------
Pesaran, M. H., Shin, Y., & Smith, R. (2001).
Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Optional, Union, List

from .nardl import NARDL, NARDLResults
from .utils import (
    FormulaParser, level_lag_name, pos_name, neg_name,
    significance_stars
)


class ECMResults:
    """
    Container for ECM estimation results.

    Attributes
    ----------
    long_run_model : statsmodels OLS results (long-run equation)
    short_run_model : NARDLResults (short-run equation with ECM term)
    ecm_coefficient : float (speed of adjustment)
    case : int
    """

    def __init__(self, long_run_model, short_run_results,
                 ecm_coefficient, ecm_residuals, case, notes=None):
        self.long_run_model = long_run_model
        self.short_run = short_run_results
        self.ecm_coefficient = ecm_coefficient
        self.ecm_residuals = ecm_residuals
        self.case = case
        self.notes = notes or []

        # Validate ECM coefficient
        if ecm_coefficient >= 0:
            self.notes.append(
                "⚠ ECM coefficient is non-negative — may indicate no "
                "long-run equilibrium adjustment.")
        if ecm_coefficient < -1:
            self.notes.append(
                "⚠ ECM coefficient < -1 — may suggest over-adjustment "
                "or instability.")

    def summary(self, print_output: bool = True) -> str:
        """Print ECM summary."""
        lines = []
        lines.append("=" * 78)
        lines.append("          ERROR CORRECTION MODEL (ECM) RESULTS")
        lines.append("=" * 78)
        lines.append(f"  Case            : {self.case}")
        lines.append(f"  ECM Coefficient : {self.ecm_coefficient:.6f}")

        if self.ecm_coefficient < 0 and self.ecm_coefficient > -1:
            half_life = np.log(0.5) / np.log(1 + self.ecm_coefficient)
            lines.append(f"  Half-life       : {half_life:.1f} periods")

        lines.append("-" * 78)

        # Long-run equation
        lines.append("\n  LONG-RUN EQUATION")
        lines.append("-" * 78)
        lr_summary = self.long_run_model.summary2().tables[1]
        lines.append(str(lr_summary))

        # Short-run equation
        lines.append("\n  SHORT-RUN EQUATION (with ECM term)")
        lines.append("-" * 78)
        self.short_run.summary(print_output=False)
        lines.append(self.short_run.summary(print_output=False))

        # Notes
        if self.notes:
            lines.append("\n  NOTES:")
            for note in self.notes:
                lines.append(f"    {note}")

        lines.append("=" * 78)

        text = "\n".join(lines)
        if print_output:
            print(text)
        return text


def estimate_ecm(
    data: pd.DataFrame,
    formula: str,
    case: int = 3,
    maxlag: int = 4,
    criterion: str = 'BIC',
    mode: Union[str, Dict, List] = 'grid',
    verbose: bool = False,
) -> ECMResults:
    """
    Estimate Error Correction Model (ECM).

    Two-step procedure:
    1. Estimate long-run equilibrium: y_{t-1} ~ x_{t-1} (+ const/trend)
    2. Include lagged residuals (ECM term) in short-run NARDL

    Parameters
    ----------
    data : pd.DataFrame
    formula : str
    case : int, 1-5
    maxlag : int
    criterion : str
    mode : str, dict, or list
    verbose : bool

    Returns
    -------
    ECMResults
    """
    from .prepare import prepare_nardl_data

    parsed = FormulaParser.parse(formula)
    dep_var = parsed['dep_var']
    indep_vars = parsed['indep_vars']
    all_lr_asym = parsed['all_asym_lr']

    notes = []

    # Auto-adjust case
    if parsed['no_constant']:
        if case > 1:
            notes.append("No constant in model — case adjusted to 1.")
        case = 1
    elif parsed['trend'] and case < 4:
        notes.append("Trend detected — case adjusted to 5.")
        case = 5

    # ── Step 1: Long-run equation ────────────────────────────────
    # Build long-run variable names
    lr_dep = level_lag_name(dep_var)
    lr_indep = []

    for var in indep_vars:
        if var in all_lr_asym:
            lr_indep.append(level_lag_name(pos_name(var)))
            lr_indep.append(level_lag_name(neg_name(var)))
        else:
            lr_indep.append(level_lag_name(var))

    # Prepare data with all variables
    prep = prepare_nardl_data(data, formula, maxlag)
    ext_data = prep['data']

    # Build long-run regression
    lr_cols = [lr_dep] + lr_indep
    for col in lr_cols:
        if col not in ext_data.columns:
            raise ValueError(f"Column '{col}' not found. Check formula.")

    lr_data = ext_data[lr_cols].dropna()
    y_lr = lr_data[lr_dep].values
    X_lr = lr_data[lr_indep].values

    # Add constant/trend based on case
    lr_col_names = list(lr_indep)
    if case == 2:
        X_lr = sm.add_constant(X_lr)
        lr_col_names = ['const'] + lr_col_names
    elif case == 4:
        trend = np.arange(1, len(lr_data) + 1)
        X_lr = np.column_stack([X_lr, trend])
        lr_col_names.append('trend')
        X_lr = sm.add_constant(X_lr)
        lr_col_names = ['const'] + lr_col_names  # case 4 has unrestricted const
    elif case in (3, 5):
        X_lr = sm.add_constant(X_lr)
        lr_col_names = ['const'] + lr_col_names
    # case 1: no constant, no trend

    X_lr_df = pd.DataFrame(X_lr, columns=lr_col_names, index=lr_data.index)
    lr_model = sm.OLS(y_lr, X_lr_df).fit()

    # ── Get ECM residuals ────────────────────────────────────────
    ecm_resid = np.full(len(ext_data), np.nan)
    resid_values = lr_model.resid
    lr_indices = lr_data.index

    # Lag the residuals by 1
    for i in range(1, len(lr_indices)):
        curr_idx = lr_indices[i]
        prev_idx = lr_indices[i - 1]
        if curr_idx < len(ecm_resid):
            ecm_resid[curr_idx] = resid_values.iloc[i - 1]

    # Add ECM residuals to data
    ext_data_ecm = ext_data.copy()
    ext_data_ecm['EcmRes'] = ecm_resid

    # ── Step 2: Short-run equation with ECM term ─────────────────
    # Modify formula to include EcmRes as deterministic
    ecm_formula = formula
    if 'deterministic(' in ecm_formula.lower():
        # Add EcmRes to existing deterministic
        import re
        ecm_formula = re.sub(
            r'([Dd]et(?:erministic)?)\(([^)]+)\)',
            r'\1(\2 + EcmRes)',
            ecm_formula
        )
    else:
        ecm_formula += ' + deterministic(EcmRes)'

    if case == 5 and 'trend' not in ecm_formula.lower():
        ecm_formula += ' + trend'

    # Fit short-run NARDL
    sr_nardl = NARDL(
        data=ext_data_ecm,
        formula=ecm_formula,
        maxlag=maxlag,
        criterion=criterion,
        case=case,
    )
    sr_results = sr_nardl.fit(mode=mode, verbose=verbose)

    # Extract ECM coefficient
    ecm_coef = sr_results.coefficients.get('EcmRes', np.nan)

    return ECMResults(
        long_run_model=lr_model,
        short_run_results=sr_results,
        ecm_coefficient=ecm_coef,
        ecm_residuals=ecm_resid,
        case=case,
        notes=notes,
    )
