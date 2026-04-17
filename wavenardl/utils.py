"""
Shared utilities for the wavenardl package.

Provides helper functions for formula parsing, variable naming,
significance formatting, and common mathematical operations.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any


# ──────────────────────────────────────────────────────────────────────
# Significance helpers
# ──────────────────────────────────────────────────────────────────────

def significance_stars(pvalue: float) -> str:
    """Return significance stars based on p-value."""
    if pvalue <= 0.01:
        return "***"
    elif pvalue <= 0.05:
        return "**"
    elif pvalue <= 0.10:
        return "*"
    return ""


def format_coefficient(value: float, se: float, pvalue: float,
                       decimals: int = 4) -> str:
    """Format coefficient with standard error and significance stars."""
    stars = significance_stars(pvalue)
    return f"{value:.{decimals}f}{stars}\n({se:.{decimals}f})"


# ──────────────────────────────────────────────────────────────────────
# Formula parsing
# ──────────────────────────────────────────────────────────────────────

class FormulaParser:
    """
    Parse R-like formula strings for NARDL models.

    Supports:
        y ~ x1 + x2 + Asymmetric(x1) + Sasymmetric(x2) + 
            Lasymmetric(x3) + deterministic(d1 + d2) + trend

    Returns a structured dict of variable roles.
    """

    # Patterns for special operators (case-insensitive)
    _OPERATORS = {
        'asymmetric':   re.compile(r'\b[Aa]sym(?:metric)?\s*\(\s*([^)]+)\s*\)', re.I),
        'sasymmetric':  re.compile(r'\b[Ss]asym(?:metric)?\s*\(\s*([^)]+)\s*\)', re.I),
        'lasymmetric':  re.compile(r'\b[Ll]asym(?:metric)?\s*\(\s*([^)]+)\s*\)', re.I),
        'deterministic': re.compile(r'\b[Dd]et(?:erministic)?\s*\(\s*([^)]+)\s*\)', re.I),
    }

    @staticmethod
    def parse(formula: str) -> Dict[str, Any]:
        """
        Parse a formula string and return variable classifications.

        Returns
        -------
        dict with keys:
            dep_var : str
            indep_vars : list[str]
            asym_vars : list[str]       (both LR and SR asymmetric)
            sasym_vars : list[str]      (short-run only asymmetric)
            lasym_vars : list[str]      (long-run only asymmetric)
            deterministic : list[str]   (dummy / exogenous regressors)
            trend : bool
            no_constant : bool
        """
        result = {
            'dep_var': '',
            'indep_vars': [],
            'asym_vars': [],
            'sasym_vars': [],
            'lasym_vars': [],
            'deterministic': [],
            'trend': False,
            'no_constant': False,
        }

        if '~' not in formula:
            raise ValueError("Formula must contain '~' separating dependent and independent sides.")

        lhs, rhs = formula.split('~', 1)
        result['dep_var'] = lhs.strip()

        # Check for -1 (no constant)
        if '-1' in rhs or '- 1' in rhs:
            result['no_constant'] = True
            rhs = rhs.replace('-1', '').replace('- 1', '')

        # Check for trend
        if re.search(r'\btrend\b', rhs, re.I):
            result['trend'] = True
            rhs = re.sub(r'\btrend\b', '', rhs, flags=re.I)

        # Extract operator-wrapped variables
        for op_name, pattern in FormulaParser._OPERATORS.items():
            for match in pattern.finditer(rhs):
                vars_str = match.group(1)
                vars_list = [v.strip() for v in vars_str.replace('+', ',').split(',')
                             if v.strip()]
                if op_name == 'asymmetric':
                    result['asym_vars'].extend(vars_list)
                elif op_name == 'sasymmetric':
                    result['sasym_vars'].extend(vars_list)
                elif op_name == 'lasymmetric':
                    result['lasym_vars'].extend(vars_list)
                elif op_name == 'deterministic':
                    result['deterministic'].extend(vars_list)
            rhs = pattern.sub('', rhs)

        # Remaining terms are independent variables
        remaining = [v.strip() for v in rhs.split('+') if v.strip()]
        result['indep_vars'] = remaining

        # Deduplicate
        for key in ['asym_vars', 'sasym_vars', 'lasym_vars', 'deterministic']:
            result[key] = list(dict.fromkeys(result[key]))

        # Merge asymmetric into both S and L
        result['all_asym_lr'] = list(dict.fromkeys(
            result['asym_vars'] + result['lasym_vars']))
        result['all_asym_sr'] = list(dict.fromkeys(
            result['asym_vars'] + result['sasym_vars']))

        return result


# ──────────────────────────────────────────────────────────────────────
# Variable naming conventions
# ──────────────────────────────────────────────────────────────────────

# Prefixes / suffixes for positive and negative partial sums
ASYM_PREFIX = ('', '')
ASYM_SUFFIX = ('_POS', '_NEG')


def pos_name(var: str) -> str:
    """Return the positive partial-sum variable name."""
    return f"{ASYM_PREFIX[0]}{var}{ASYM_SUFFIX[0]}"


def neg_name(var: str) -> str:
    """Return the negative partial-sum variable name."""
    return f"{ASYM_PREFIX[1]}{var}{ASYM_SUFFIX[1]}"


def diff_name(var: str, lag: int) -> str:
    """Return the differenced-lagged variable name: D{lag}.{var}"""
    return f"D{lag}.{var}"


def level_lag_name(var: str) -> str:
    """Return the level-lagged variable name: L1.{var}"""
    return f"L1.{var}"


# ──────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────

def ensure_dataframe(data) -> pd.DataFrame:
    """Convert input to DataFrame if needed."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame(data)
    raise TypeError(f"Cannot convert {type(data)} to DataFrame.")


def compute_ic(ll: float, k: int, n: int, criterion: str = 'BIC') -> float:
    """
    Compute information criterion (normalized by n, matching R's kardl).

    Parameters
    ----------
    ll : float — log-likelihood
    k  : int  — number of parameters
    n  : int  — number of observations
    criterion : str — 'AIC', 'BIC', 'AICc', 'HQ'
    """
    criterion = criterion.upper()
    if criterion == 'AIC':
        return (2 * k - 2 * ll) / n
    elif criterion == 'BIC':
        return (np.log(n) * k - 2 * ll) / n
    elif criterion == 'AICC':
        aic = (2 * k - 2 * ll) / n
        correction = (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else np.inf
        return aic + correction
    elif criterion == 'HQ':
        return (2 * np.log(np.log(n)) * k - 2 * ll) / n
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use AIC, BIC, AICc, or HQ.")
