"""
Core NARDL / ARDL estimation class.

Implements the Nonlinear Autoregressive Distributed Lag model
following Shin, Yu & Greenwood-Nimmo (2014) and the R `kardl` package.

The general NARDL specification is:

    Δy_t = μ + ρ y_{t-1} + θ⁺ x⁺_{t-1} + θ⁻ x⁻_{t-1}
         + Σ γ_i Δy_{t-i} + Σ (β⁺_j Δx⁺_{t-j} + β⁻_j Δx⁻_{t-j}) + ε_t

References
----------
Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014). Modelling asymmetric
    cointegration and dynamic multipliers in a nonlinear ARDL framework.
    Festschrift in Honor of Peter Schmidt, 281-314.

Pesaran, M. H., Shin, Y., & Smith, R. (2001). Bounds testing approaches
    to the analysis of level relationship. JASA, 16(3), 289-326.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Tuple, Union

from .prepare import prepare_nardl_data, build_regression_data
from .lagselect import select_lags
from .utils import (
    FormulaParser, compute_ic, significance_stars,
    pos_name, neg_name, diff_name, level_lag_name
)


class NARDLResults:
    """
    Container for NARDL estimation results.

    Attributes
    ----------
    model : statsmodels OLS results
    prep : dict with prepared data info
    optimal_lags : dict mapping variable -> lag order
    lag_results : DataFrame of all evaluated lag combinations
    case : int, PSS case (1-5)
    """

    def __init__(self, model, prep, optimal_lags, lag_results,
                 case, reg_data, dep_col, indep_cols):
        self.model = model
        self.prep = prep
        self.optimal_lags = optimal_lags
        self.lag_results = lag_results
        self.case = case
        self.reg_data = reg_data
        self.dep_col = dep_col
        self.indep_cols = indep_cols

        # Extract key info
        self.nobs = int(model.nobs)
        self.nparams = len(model.params)
        self.rsquared = model.rsquared
        self.rsquared_adj = model.rsquared_adj
        self.fstatistic = model.fvalue
        self.f_pvalue = model.f_pvalue
        self.aic = model.aic
        self.bic = model.bic
        self.llf = model.llf
        self.durbin_watson = sm.stats.durbin_watson(model.resid)

        # Normalized IC (matching R's kardl)
        self.aic_norm = compute_ic(self.llf, self.nparams, self.nobs, 'AIC')
        self.bic_norm = compute_ic(self.llf, self.nparams, self.nobs, 'BIC')
        self.hq_norm = compute_ic(self.llf, self.nparams, self.nobs, 'HQ')

        # Store coefficient info
        self._build_coef_table()

    def _build_coef_table(self):
        """Build a formatted coefficient table."""
        params = self.model.params
        bse = self.model.bse
        tvalues = self.model.tvalues
        pvalues = self.model.pvalues

        names = list(self.model.params.index) if hasattr(self.model.params, 'index') \
            else [f'x{i}' for i in range(len(params))]

        self.coef_table = pd.DataFrame({
            'Coefficient': params.values if hasattr(params, 'values') else params,
            'Std.Error': bse.values if hasattr(bse, 'values') else bse,
            't-value': tvalues.values if hasattr(tvalues, 'values') else tvalues,
            'p-value': pvalues.values if hasattr(pvalues, 'values') else pvalues,
        }, index=names)

        self.coef_table['Signif'] = self.coef_table['p-value'].apply(significance_stars)

    @property
    def coefficients(self):
        """Return coefficient values as dict."""
        return dict(zip(self.coef_table.index, self.coef_table['Coefficient']))

    @property
    def residuals(self):
        """Return residuals."""
        return self.model.resid

    @property
    def fitted_values(self):
        """Return fitted values."""
        return self.model.fittedvalues

    def summary(self, print_output: bool = True) -> str:
        """
        Print a comprehensive summary of the NARDL estimation.
        """
        parsed = self.prep['parsed']
        dep = parsed['dep_var']
        lines = []

        lines.append("=" * 78)
        lines.append("            NARDL MODEL ESTIMATION RESULTS")
        lines.append("=" * 78)

        # Model info
        model_type = "NARDL" if (parsed['all_asym_lr'] or parsed['all_asym_sr']) else "ARDL"
        lines.append(f"  Model Type      : {model_type}")
        lines.append(f"  Dependent Var   : D.{dep}")
        lines.append(f"  Observations    : {self.nobs}")
        lines.append(f"  Parameters      : {self.nparams}")
        lines.append(f"  Optimal Lags    : {self.optimal_lags}")
        lines.append(f"  Case            : {self.case}")
        lines.append("-" * 78)

        # Fit statistics
        lines.append(f"  R-squared       : {self.rsquared:.6f}")
        lines.append(f"  Adj. R-squared  : {self.rsquared_adj:.6f}")
        lines.append(f"  F-statistic     : {self.fstatistic:.4f}  (p = {self.f_pvalue:.6f})")
        lines.append(f"  Log-Likelihood  : {self.llf:.4f}")
        lines.append(f"  AIC             : {self.aic:.4f}")
        lines.append(f"  BIC             : {self.bic:.4f}")
        lines.append(f"  Durbin-Watson   : {self.durbin_watson:.4f}")
        lines.append("-" * 78)

        # Coefficients
        lines.append("")
        lines.append("  COEFFICIENT ESTIMATES")
        lines.append("-" * 78)
        lines.append(f"  {'Variable':<25s} {'Coef':>10s} {'Std.Err':>10s} "
                     f"{'t-value':>10s} {'p-value':>10s} {'':>5s}")
        lines.append("-" * 78)

        # Separate into long-run and short-run
        lr_names = self.prep['long_run_vars']
        for idx, row in self.coef_table.iterrows():
            section = "LR" if idx in lr_names else "SR"
            lines.append(
                f"  {idx:<25s} {row['Coefficient']:>10.4f} {row['Std.Error']:>10.4f} "
                f"{row['t-value']:>10.4f} {row['p-value']:>10.4f} {row['Signif']:>5s}"
            )

        lines.append("-" * 78)
        lines.append("  Signif. codes: '***' 0.01, '**' 0.05, '*' 0.10")
        lines.append("=" * 78)

        text = "\n".join(lines)
        if print_output:
            print(text)
        return text

    def predict(self, newdata: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate predictions (fitted values or out-of-sample)."""
        if newdata is None:
            return self.fitted_values
        raise NotImplementedError("Out-of-sample prediction not yet implemented.")


class NARDL:
    """
    Nonlinear Autoregressive Distributed Lag (NARDL) Model.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing all variables.
    formula : str
        Model specification, e.g.:
        "y ~ x1 + x2 + Asymmetric(x1) + deterministic(d1) + trend"
    maxlag : int
        Maximum lag order to consider (default: 4).
    criterion : str
        Information criterion for lag selection: 'AIC', 'BIC', 'AICc', 'HQ'.
    case : int
        PSS case for deterministic specification (1-5, default: 3).
    different_asym_lag : bool
        Allow different lags for positive and negative components.

    Examples
    --------
    >>> model = NARDL(data, "oil ~ exrate + Asymmetric(exrate)", maxlag=4)
    >>> results = model.fit(mode='grid')
    >>> results.summary()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        maxlag: int = 4,
        criterion: str = 'BIC',
        case: int = 3,
        different_asym_lag: bool = False,
    ):
        self.data = data
        self.formula = formula
        self.maxlag = maxlag
        self.criterion = criterion
        self.case = case
        self.different_asym_lag = different_asym_lag

        # Prepare data
        self.prep = prepare_nardl_data(
            data=data,
            formula=formula,
            maxlag=maxlag,
            different_asym_lag=different_asym_lag,
        )

        # Auto-detect case based on formula
        parsed = self.prep['parsed']
        if parsed['no_constant']:
            self.case = 1
        elif parsed['trend'] and case < 4:
            self.case = 5

    def fit(
        self,
        mode: Union[str, Dict, List] = 'grid',
        verbose: bool = False,
    ) -> NARDLResults:
        """
        Estimate the NARDL model.

        Parameters
        ----------
        mode : str or dict or list
            'grid' — exhaustive search (default)
            'quick' — sequential search
            dict or list — user-defined lag vector
        verbose : bool
            Print progress information.

        Returns
        -------
        NARDLResults
        """
        add_constant = not self.prep['parsed']['no_constant']

        if verbose:
            print(f"Selecting optimal lags (mode={mode}, criterion={self.criterion})...")

        # Select lags
        optimal_lags, lag_results = select_lags(
            prep=self.prep,
            mode=mode,
            criterion=self.criterion,
            add_constant=add_constant,
            verbose=verbose,
        )

        if verbose:
            print(f"Optimal lags: {optimal_lags}")
            print("Estimating final model...")

        # Build and estimate final model
        reg_data, dep_col, indep_cols = build_regression_data(self.prep, optimal_lags)

        y = reg_data[dep_col].values
        X = reg_data[indep_cols].values

        col_names = list(indep_cols)
        if add_constant:
            X = sm.add_constant(X)
            col_names = ['const'] + col_names

        ols = sm.OLS(y, X, missing='drop')
        model = ols.fit()

        # Rename parameters
        model_params = pd.Series(model.params, index=col_names)
        model._results.params = model_params.values

        # Create a proper indexed results object
        # We need to rebuild with proper names
        X_df = pd.DataFrame(X, columns=col_names)
        y_series = pd.Series(y, name=dep_col)
        model = sm.OLS(y_series, X_df, missing='drop').fit()

        results = NARDLResults(
            model=model,
            prep=self.prep,
            optimal_lags=optimal_lags,
            lag_results=lag_results,
            case=self.case,
            reg_data=reg_data,
            dep_col=dep_col,
            indep_cols=indep_cols,
        )

        if verbose:
            print("Estimation complete.\n")

        return results
