"""
Data preparation and asymmetric decomposition for NARDL models.

Implements partial sum decomposition of Shin et al. (2013) and
creates all necessary differenced/lagged variables for estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .utils import (
    FormulaParser, pos_name, neg_name, diff_name, level_lag_name,
    ensure_dataframe
)


def partial_sum_decomposition(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a variable X into positive and negative partial sums.

    X_pos_t = Σ_{j=1}^{t} max(ΔX_j, 0)
    X_neg_t = Σ_{j=1}^{t} min(ΔX_j, 0)

    Parameters
    ----------
    series : array-like
        The original variable in levels.

    Returns
    -------
    x_pos, x_neg : np.ndarray
        Positive and negative partial sums (same length as input, first value = NaN).

    References
    ----------
    Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014).
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    dx = np.diff(x)  # length n-1

    x_pos = np.full(n, np.nan)
    x_neg = np.full(n, np.nan)

    cum_pos = 0.0
    cum_neg = 0.0

    for i in range(len(dx)):
        if dx[i] > 0:
            cum_pos += dx[i]
        elif dx[i] < 0:
            cum_neg += dx[i]
        # If dx[i] == 0, both remain unchanged
        x_pos[i + 1] = cum_pos
        x_neg[i + 1] = cum_neg

    return x_pos, x_neg


def prepare_nardl_data(
    data: pd.DataFrame,
    formula: str,
    maxlag: int = 4,
    different_asym_lag: bool = False,
) -> Dict:
    """
    Prepare data for NARDL estimation.

    Steps:
    1. Parse formula to identify variable roles
    2. Create asymmetric partial sums (X_POS, X_NEG) for asymmetric variables
    3. Create differenced variables (ΔY, ΔX) and their lags
    4. Create level-lagged variables (Y_{t-1}, X_{t-1}) for long-run
    5. Add trend if requested

    Parameters
    ----------
    data : pd.DataFrame
    formula : str
    maxlag : int
    different_asym_lag : bool
        If True, allow different lag lengths for POS and NEG components.

    Returns
    -------
    dict with keys:
        'parsed' : parsed formula dict
        'data' : pd.DataFrame with all constructed variables
        'dep_diff' : str, name of D0.y (dependent in ARDL equation)
        'long_run_vars' : list[str], level-lagged variable names
        'short_run_vars' : list[str], names in short-run part
        'deterministic_vars' : list[str]
    """
    data = ensure_dataframe(data)
    parsed = FormulaParser.parse(formula)

    dep_var = parsed['dep_var']
    indep_vars = parsed['indep_vars']
    all_lr_asym = parsed['all_asym_lr']
    all_sr_asym = parsed['all_asym_sr']
    all_asym = list(dict.fromkeys(all_lr_asym + all_sr_asym))
    deterministic = parsed['deterministic']

    # Validate variables exist in data
    all_model_vars = [dep_var] + indep_vars
    for v in all_model_vars:
        if v not in data.columns:
            raise ValueError(f"Variable '{v}' not found in data. "
                             f"Available: {list(data.columns)}")
    for v in deterministic:
        if v not in data.columns:
            raise ValueError(f"Deterministic variable '{v}' not found in data.")
    for v in all_asym:
        if v not in indep_vars:
            raise ValueError(
                f"Asymmetric variable '{v}' must also appear as an independent variable.")

    # ── Build extended DataFrame ────────────────────────────────────
    ext = data.copy()

    # 1. Create partial sums for asymmetric variables
    for v in all_asym:
        pn = pos_name(v)
        nn = neg_name(v)
        if pn not in ext.columns:
            x_pos, x_neg = partial_sum_decomposition(ext[v].values)
            ext[pn] = x_pos
            ext[nn] = x_neg

    # 2. Identify short-run and long-run variable lists
    #    Short-run: replace SR-asymmetric vars with their POS/NEG counterparts
    sr_vars_list = [dep_var]
    for v in indep_vars:
        if v in all_sr_asym:
            sr_vars_list.extend([pos_name(v), neg_name(v)])
        else:
            sr_vars_list.append(v)

    #    Long-run: replace LR-asymmetric vars with their POS/NEG counterparts
    lr_vars_list = [dep_var]
    for v in indep_vars:
        if v in all_lr_asym:
            lr_vars_list.extend([pos_name(v), neg_name(v)])
        else:
            lr_vars_list.append(v)

    # 3. Create differenced variables and their lags (for short-run)
    for v in sr_vars_list:
        values = ext[v].values.astype(float)
        dv = np.full(len(values), np.nan)
        dv[1:] = np.diff(values)

        for lag in range(0, maxlag + 1):
            col_name = diff_name(v, lag)
            if col_name not in ext.columns:
                lagged = np.full(len(values), np.nan)
                start = lag + 1
                if start < len(values):
                    lagged[start:] = dv[1:len(values) - lag]
                ext[col_name] = lagged

    # 4. Create level-lagged variables (for long-run)
    for v in lr_vars_list:
        col_name = level_lag_name(v)
        if col_name not in ext.columns:
            values = ext[v].values.astype(float)
            lagged = np.full(len(values), np.nan)
            lagged[1:] = values[:-1]
            ext[col_name] = lagged

    # 5. Add trend if requested
    if parsed['trend']:
        if 'trend' not in ext.columns:
            ext['trend'] = np.arange(1, len(ext) + 1)
        if 'trend' not in deterministic:
            deterministic = deterministic + ['trend']

    # ── Build name lists for the regression ─────────────────────────
    dep_diff_name = diff_name(dep_var, 0)

    # Long-run variable names (level-lagged)
    long_run_names = [level_lag_name(v) for v in lr_vars_list]

    # Short-run variable names (differenced-lagged)
    # Dependent: lags 1..maxlag;  Independent: lags 0..maxlag
    short_run_names_by_var = {}
    for idx, v in enumerate(sr_vars_list):
        start_lag = 1 if idx == 0 else 0  # dep starts at lag 1
        var_lags = [diff_name(v, lag) for lag in range(start_lag, maxlag + 1)]
        short_run_names_by_var[v] = var_lags

    # Flatten
    all_short_run_names = []
    for v in sr_vars_list:
        all_short_run_names.extend(short_run_names_by_var[v])

    return {
        'parsed': parsed,
        'data': ext,
        'dep_var': dep_var,
        'dep_diff': dep_diff_name,
        'long_run_vars': long_run_names,
        'long_run_base_vars': lr_vars_list,
        'short_run_vars': all_short_run_names,
        'short_run_by_var': short_run_names_by_var,
        'short_run_base_vars': sr_vars_list,
        'deterministic_vars': deterministic,
        'maxlag': maxlag,
        'different_asym_lag': different_asym_lag,
    }


def build_regression_data(
    prep: Dict,
    lag_vector: Dict[str, int],
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Build the final regression DataFrame for a specific lag configuration.

    Parameters
    ----------
    prep : dict from prepare_nardl_data
    lag_vector : dict mapping each short-run variable to its lag order
                 e.g. {'y': 2, 'x_POS': 1, 'x_NEG': 3}

    Returns
    -------
    reg_data : pd.DataFrame (rows with NaN dropped)
    dep_col : str (name of dependent variable column)
    indep_cols : list[str] (names of all regressors)
    """
    data = prep['data']
    dep_col = prep['dep_diff']

    # Collect all regressor columns
    indep_cols = []

    # Long-run (level-lagged) variables
    indep_cols.extend(prep['long_run_vars'])

    # Short-run (differenced-lagged) variables based on lag_vector
    sr_vars = prep['short_run_base_vars']
    for idx, v in enumerate(sr_vars):
        start_lag = 1 if idx == 0 else 0
        max_v_lag = lag_vector.get(v, prep['maxlag'])
        for lag in range(start_lag, max_v_lag + 1):
            col = diff_name(v, lag)
            if col in data.columns:
                indep_cols.append(col)

    # Deterministic variables
    indep_cols.extend(prep['deterministic_vars'])

    # Build regression data
    all_cols = [dep_col] + indep_cols
    missing = [c for c in all_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    reg_data = data[all_cols].dropna()

    return reg_data, dep_col, indep_cols
