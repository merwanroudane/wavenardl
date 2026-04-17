"""
Automatic lag selection for ARDL / NARDL models.

Supports grid search, quick search, and user-defined lag vectors.
Criteria: AIC, BIC, AICc, HQ (normalized by n, matching R's kardl).
"""

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Optional, Tuple, Callable, Union
from .utils import compute_ic
from .prepare import build_regression_data, diff_name


def _estimate_for_lag(
    prep: Dict,
    lag_vector: Dict[str, int],
    add_constant: bool = True,
) -> Optional[Dict]:
    """
    Estimate OLS for a specific lag configuration and return IC values.

    Returns None if estimation fails (e.g., singular matrix).
    """
    try:
        reg_data, dep_col, indep_cols = build_regression_data(prep, lag_vector)
        y = reg_data[dep_col].values
        X = reg_data[indep_cols].values

        if add_constant and not prep['parsed'].get('no_constant', False):
            X = sm.add_constant(X)

        if X.shape[0] <= X.shape[1] + 1:
            return None  # Not enough observations

        model = sm.OLS(y, X, missing='drop').fit()
        n = model.nobs
        k = len(model.params)
        ll = model.llf

        return {
            'lag_vector': lag_vector.copy(),
            'AIC': compute_ic(ll, k, n, 'AIC'),
            'BIC': compute_ic(ll, k, n, 'BIC'),
            'AICc': compute_ic(ll, k, n, 'AICc'),
            'HQ': compute_ic(ll, k, n, 'HQ'),
            'n': n,
            'k': k,
            'll': ll,
        }
    except Exception:
        return None


def grid_search(
    prep: Dict,
    criterion: str = 'BIC',
    add_constant: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, int], pd.DataFrame]:
    """
    Exhaustive grid search over all lag combinations.

    Parameters
    ----------
    prep : dict from prepare_nardl_data
    criterion : str, one of 'AIC', 'BIC', 'AICc', 'HQ'
    add_constant : bool
    verbose : bool

    Returns
    -------
    best_lags : dict mapping variable -> optimal lag
    results_df : DataFrame with all evaluated lag combinations and their IC values
    """
    maxlag = prep['maxlag']
    sr_vars = prep['short_run_base_vars']

    # Generate all lag combinations
    # Dependent variable: 1..maxlag; Independent variables: 0..maxlag
    ranges = []
    var_names = []
    for idx, v in enumerate(sr_vars):
        if idx == 0:
            ranges.append(range(1, maxlag + 1))  # dep: at least 1 lag
        else:
            ranges.append(range(0, maxlag + 1))  # indep: 0 means contemporaneous only
        var_names.append(v)

    results = []
    total = 1
    for r in ranges:
        total *= len(r)

    evaluated = 0
    for combo in itertools.product(*ranges):
        lag_vector = dict(zip(var_names, combo))
        result = _estimate_for_lag(prep, lag_vector, add_constant)
        if result is not None:
            results.append(result)
        evaluated += 1
        if verbose and evaluated % 50 == 0:
            print(f"  Evaluated {evaluated}/{total} lag combinations...")

    if not results:
        raise RuntimeError("No valid lag combination found. Check data and formula.")

    results_df = pd.DataFrame(results)
    criterion = criterion.upper()
    if criterion not in results_df.columns:
        raise ValueError(f"Criterion '{criterion}' not available.")

    best_idx = results_df[criterion].idxmin()
    best_lags = results_df.loc[best_idx, 'lag_vector']

    if verbose:
        print(f"\n  Best lags by {criterion}: {best_lags}")
        print(f"  {criterion} = {results_df.loc[best_idx, criterion]:.6f}")

    return best_lags, results_df


def quick_search(
    prep: Dict,
    criterion: str = 'BIC',
    add_constant: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, int], pd.DataFrame]:
    """
    Sequential (stepwise) lag selection — faster than grid search.

    Selects optimal lag for each variable sequentially while holding
    others at their current best.
    """
    maxlag = prep['maxlag']
    sr_vars = prep['short_run_base_vars']
    criterion = criterion.upper()

    # Start with all lags = 1
    current_best = {}
    for idx, v in enumerate(sr_vars):
        current_best[v] = 1

    results = []
    improved = True
    iteration = 0

    while improved and iteration < 10:
        improved = False
        iteration += 1
        if verbose:
            print(f"  Quick search iteration {iteration}...")

        for idx, v in enumerate(sr_vars):
            best_ic = np.inf
            best_lag = current_best[v]
            start = 1 if idx == 0 else 0

            for lag in range(start, maxlag + 1):
                test_lags = current_best.copy()
                test_lags[v] = lag
                result = _estimate_for_lag(prep, test_lags, add_constant)
                if result is not None:
                    results.append(result)
                    if result[criterion] < best_ic:
                        best_ic = result[criterion]
                        best_lag = lag

            if best_lag != current_best[v]:
                current_best[v] = best_lag
                improved = True

    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    return current_best, results_df


def select_lags(
    prep: Dict,
    mode: Union[str, Dict[str, int], List[int]] = 'grid',
    criterion: str = 'BIC',
    add_constant: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, int], pd.DataFrame]:
    """
    Select optimal lag structure.

    Parameters
    ----------
    prep : dict from prepare_nardl_data
    mode : str ('grid', 'quick') or dict/list (user-defined lags)
    criterion : str
    add_constant : bool
    verbose : bool

    Returns
    -------
    best_lags : dict
    results_df : DataFrame
    """
    if isinstance(mode, dict):
        # User-defined lag dictionary
        return mode, pd.DataFrame()
    elif isinstance(mode, (list, tuple, np.ndarray)):
        # User-defined lag vector
        sr_vars = prep['short_run_base_vars']
        if len(mode) != len(sr_vars):
            raise ValueError(
                f"Lag vector length ({len(mode)}) must match number of "
                f"short-run variables ({len(sr_vars)}): {sr_vars}")
        lag_dict = dict(zip(sr_vars, mode))
        return lag_dict, pd.DataFrame()
    elif isinstance(mode, str):
        mode = mode.lower()
        if mode == 'grid':
            return grid_search(prep, criterion, add_constant, verbose)
        elif mode == 'quick':
            return quick_search(prep, criterion, add_constant, verbose)
        else:
            raise ValueError(f"Unknown mode: '{mode}'. Use 'grid', 'quick', or provide a lag vector.")
    else:
        raise TypeError(f"mode must be str, dict, or list, not {type(mode)}")
