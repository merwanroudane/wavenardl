"""
Critical value tables for PSS bounds tests and Narayan small-sample tests.

References
----------
Pesaran, M. H., Shin, Y., & Smith, R. (2001). Bounds testing approaches
    to the analysis of level relationship. JASA, 16(3), 289-326.

Narayan, P. K. (2005). The saving and investment nexus for China.
    Applied Economics, 37(17), 1979-1990.
"""

import numpy as np
from typing import Dict, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════
# PSS (2001) Critical Values — Asymptotic
# Tables CI–CV for Cases I–V
# Format: For each case, a flat array ordered as:
#   [k=0: 10%L, 10%U, 5%L, 5%U, 2.5%L, 2.5%U, 1%L, 1%U,
#    k=1: ..., k=2: ..., ..., k=10: ...]
# ══════════════════════════════════════════════════════════════════════

PSS_CRITICAL_VALUES = {
    # Case I: No intercept, no trend
    1: {
        0:  (3.00, 3.00, 4.20, 4.20, 5.47, 5.47, 7.17, 7.17),
        1:  (2.44, 3.28, 3.15, 4.11, 3.88, 4.92, 4.81, 6.02),
        2:  (2.17, 3.19, 2.72, 3.83, 3.22, 4.50, 3.88, 5.30),
        3:  (2.01, 3.10, 2.45, 3.63, 2.87, 4.16, 3.42, 4.84),
        4:  (1.90, 3.01, 2.26, 3.48, 2.62, 3.90, 3.07, 4.44),
        5:  (1.81, 2.93, 2.14, 3.34, 2.44, 3.71, 2.82, 4.21),
        6:  (1.75, 2.87, 2.04, 3.24, 2.32, 3.59, 2.66, 4.05),
        7:  (1.70, 2.83, 1.97, 3.18, 2.22, 3.49, 2.54, 3.91),
        8:  (1.66, 2.79, 1.91, 3.11, 2.15, 3.40, 2.45, 3.79),
        9:  (1.63, 2.75, 1.86, 3.05, 2.08, 3.33, 2.34, 3.68),
        10: (1.60, 2.72, 1.82, 2.99, 2.02, 3.27, 2.26, 3.60),
    },
    # Case II: Restricted intercept, no trend
    2: {
        0:  (3.80, 3.80, 4.60, 4.60, 5.39, 5.39, 6.44, 6.44),
        1:  (3.02, 3.51, 3.62, 4.16, 4.18, 4.79, 4.94, 5.58),
        2:  (2.63, 3.35, 3.10, 3.87, 3.55, 4.38, 4.13, 5.00),
        3:  (2.37, 3.20, 2.79, 3.67, 3.15, 4.08, 3.65, 4.66),
        4:  (2.20, 3.09, 2.56, 3.49, 2.88, 3.87, 3.29, 4.37),
        5:  (2.08, 3.00, 2.39, 3.38, 2.70, 3.73, 3.06, 4.15),
        6:  (1.99, 2.94, 2.27, 3.28, 2.55, 3.61, 2.88, 3.99),
        7:  (1.92, 2.89, 2.17, 3.21, 2.43, 3.51, 2.73, 3.90),
        8:  (1.85, 2.85, 2.11, 3.15, 2.33, 3.42, 2.62, 3.77),
        9:  (1.80, 2.80, 2.04, 3.08, 2.24, 3.35, 2.50, 3.68),
        10: (1.76, 2.77, 1.98, 3.04, 2.18, 3.28, 2.41, 3.61),
    },
    # Case III: Unrestricted intercept, no trend
    3: {
        0:  (6.58, 6.58, 8.21, 8.21, 9.80, 9.80, 11.79, 11.79),
        1:  (4.04, 4.78, 4.94, 5.73, 5.77, 6.68, 6.84, 7.84),
        2:  (3.17, 4.14, 3.79, 4.85, 4.41, 5.52, 5.15, 6.36),
        3:  (2.72, 3.77, 3.23, 4.35, 3.69, 4.89, 4.29, 5.61),
        4:  (2.45, 3.52, 2.86, 4.01, 3.25, 4.49, 3.74, 5.06),
        5:  (2.26, 3.35, 2.62, 3.79, 2.96, 4.18, 3.41, 4.68),
        6:  (2.12, 3.23, 2.45, 3.61, 2.75, 3.99, 3.15, 4.43),
        7:  (2.03, 3.13, 2.32, 3.50, 2.60, 3.84, 2.96, 4.26),
        8:  (1.95, 3.06, 2.22, 3.39, 2.48, 3.70, 2.79, 4.10),
        9:  (1.88, 2.99, 2.14, 3.30, 2.37, 3.60, 2.65, 3.97),
        10: (1.83, 2.94, 2.06, 3.24, 2.28, 3.50, 2.54, 3.86),
    },
    # Case IV: Unrestricted intercept, restricted trend
    4: {
        0:  (5.37, 5.37, 6.29, 6.29, 7.14, 7.14, 8.26, 8.26),
        1:  (4.05, 4.49, 4.68, 5.15, 5.30, 5.83, 6.10, 6.73),
        2:  (3.38, 4.02, 3.88, 4.61, 4.37, 5.16, 4.99, 5.85),
        3:  (2.97, 3.74, 3.38, 4.23, 3.80, 4.68, 4.30, 5.23),
        4:  (2.68, 3.53, 3.05, 3.97, 3.40, 4.36, 3.81, 4.92),
        5:  (2.49, 3.38, 2.81, 3.76, 3.11, 4.13, 3.50, 4.63),
        6:  (2.33, 3.25, 2.63, 3.62, 2.90, 3.94, 3.27, 4.39),
        7:  (2.22, 3.17, 2.50, 3.50, 2.76, 3.81, 3.07, 4.23),
        8:  (2.13, 3.09, 2.38, 3.41, 2.62, 3.70, 2.93, 4.06),
        9:  (2.05, 3.02, 2.30, 3.33, 2.52, 3.60, 2.79, 3.93),
        10: (1.98, 2.97, 2.21, 3.25, 2.42, 3.52, 2.68, 3.84),
    },
    # Case V: Unrestricted intercept, unrestricted trend
    5: {
        0:  (9.81, 9.81, 11.64, 11.64, 13.36, 13.36, 15.73, 15.73),
        1:  (5.59, 6.26, 6.56, 7.30, 7.46, 8.27, 8.74, 9.63),
        2:  (4.19, 5.06, 4.87, 5.85, 5.49, 6.59, 6.34, 7.52),
        3:  (3.47, 4.45, 4.01, 5.07, 4.52, 5.62, 5.17, 6.36),
        4:  (3.03, 4.06, 3.47, 4.57, 3.89, 5.07, 4.40, 5.72),
        5:  (2.75, 3.79, 3.12, 4.25, 3.47, 4.67, 3.93, 5.23),
        6:  (2.53, 3.59, 2.87, 4.00, 3.19, 4.38, 3.60, 4.90),
        7:  (2.38, 3.45, 2.69, 3.83, 2.98, 4.16, 3.34, 4.63),
        8:  (2.26, 3.34, 2.55, 3.68, 2.82, 4.02, 3.15, 4.43),
        9:  (2.16, 3.24, 2.43, 3.56, 2.67, 3.87, 2.97, 4.24),
        10: (2.07, 3.16, 2.33, 3.46, 2.56, 3.76, 2.84, 4.10),
    },
}

# Significance level mapping for PSS
PSS_SIG_LEVELS = {
    '0.10': (0, 1),  # indices into the 8-tuple
    '0.05': (2, 3),
    '0.025': (4, 5),
    '0.01': (6, 7),
}


def get_pss_critical_values(
    case: int, k: int, sig: str = '0.05'
) -> Tuple[float, float]:
    """
    Get PSS (2001) critical values.

    Parameters
    ----------
    case : int, 1-5
    k : int, number of independent variables (0-10)
    sig : str, '0.10', '0.05', '0.025', '0.01'

    Returns
    -------
    (lower_bound, upper_bound)
    """
    if case not in PSS_CRITICAL_VALUES:
        raise ValueError(f"Invalid case {case}. Must be 1-5.")
    k = min(k, 10)
    if k not in PSS_CRITICAL_VALUES[case]:
        raise ValueError(f"k={k} not available. Must be 0-10.")
    if sig not in PSS_SIG_LEVELS:
        raise ValueError(f"Significance level '{sig}' not available.")

    vals = PSS_CRITICAL_VALUES[case][k]
    i_l, i_u = PSS_SIG_LEVELS[sig]
    return vals[i_l], vals[i_u]


def get_all_pss_critical_values(case: int, k: int) -> Dict[str, Tuple[float, float]]:
    """Get all significance levels for a given case and k."""
    result = {}
    for sig in PSS_SIG_LEVELS:
        result[sig] = get_pss_critical_values(case, k, sig)
    return result


def pss_decision(
    f_stat: float, case: int, k: int, sig: str = 'auto'
) -> Dict:
    """
    Make cointegration decision based on PSS bounds test.

    Returns dict with: decision, significance, bounds, numeric_decision
    """
    k = min(k, 10)
    all_cvs = get_all_pss_critical_values(case, k)

    if sig == 'auto':
        for s in ['0.01', '0.025', '0.05', '0.10']:
            lower, upper = all_cvs[s]
            if f_stat >= upper:
                pct = {'0.01': '1%', '0.025': '2.5%', '0.05': '5%', '0.10': '10%'}[s]
                return {
                    'decision': f'Reject H₀ → Cointegration (at {pct} level)',
                    'significance': s,
                    'bounds': (lower, upper),
                    'numeric': 1,
                }
        # Check inconclusive
        lower_10, upper_10 = all_cvs['0.10']
        if f_stat >= lower_10:
            return {
                'decision': 'Inconclusive',
                'significance': '',
                'bounds': (lower_10, upper_10),
                'numeric': 0,
            }
        return {
            'decision': 'Fail to reject H₀ → No Cointegration',
            'significance': '',
            'bounds': (lower_10, upper_10),
            'numeric': -1,
        }
    else:
        lower, upper = all_cvs[sig]
        if f_stat >= upper:
            return {
                'decision': f'Reject H₀ → Cointegration',
                'significance': sig,
                'bounds': (lower, upper),
                'numeric': 1,
            }
        elif f_stat >= lower:
            return {
                'decision': 'Inconclusive',
                'significance': sig,
                'bounds': (lower, upper),
                'numeric': 0,
            }
        else:
            return {
                'decision': 'Fail to reject H₀ → No Cointegration',
                'significance': sig,
                'bounds': (lower, upper),
                'numeric': -1,
            }
