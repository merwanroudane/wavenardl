"""
Wavelet-based NARDL model combining wavelet denoising with NARDL estimation.

Implements the full W-NARDL pipeline from Jammazi, Lahiani & Nguyen (2015):
1. Apply HTW wavelet denoising to raw data
2. Estimate NARDL on denoised series
3. Compare with standard NARDL on original series

References
----------
Jammazi, R., Lahiani, A., & Nguyen, D. K. (2015).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from .nardl import NARDL, NARDLResults
from .wavelet import HaarATrousWavelet, PyWaveletDenoiser, denoise_series


class WaveletNARDL:
    """
    Wavelet-based Nonlinear ARDL (W-NARDL) Model.

    Combines wavelet denoising with NARDL estimation per Jammazi et al. (2015).
    The smoothed (denoised) series removes noise and extreme movements,
    allowing the NARDL model to capture the true underlying relationships.

    Parameters
    ----------
    data : pd.DataFrame
    formula : str
        NARDL formula specification.
    maxlag : int
    criterion : str
    case : int
    wavelet_method : str
        'htw' (Haar à Trous, paper method), 'swt', or 'dwt'.
    wavelet : str
        Wavelet name for PyWavelets methods (default: 'haar').
    n_levels : int or None
        Wavelet decomposition levels.
    threshold_method : str
        'soft' or 'hard'.
    denoise_dep : bool
        Denoise the dependent variable (default: True).
    denoise_indep : bool
        Denoise the independent variables (default: True).

    Examples
    --------
    >>> wmodel = WaveletNARDL(
    ...     data, "oil ~ exrate + Asymmetric(exrate)",
    ...     maxlag=4, wavelet_method='htw'
    ... )
    >>> results = wmodel.fit()
    >>> results['comparison'].summary()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        maxlag: int = 4,
        criterion: str = 'BIC',
        case: int = 3,
        wavelet_method: str = 'htw',
        wavelet: str = 'haar',
        n_levels: Optional[int] = None,
        threshold_method: str = 'soft',
        denoise_dep: bool = True,
        denoise_indep: bool = True,
        different_asym_lag: bool = False,
    ):
        self.data = data.copy()
        self.formula = formula
        self.maxlag = maxlag
        self.criterion = criterion
        self.case = case
        self.wavelet_method = wavelet_method
        self.wavelet = wavelet
        self.n_levels = n_levels
        self.threshold_method = threshold_method
        self.denoise_dep = denoise_dep
        self.denoise_indep = denoise_indep
        self.different_asym_lag = different_asym_lag

        from .utils import FormulaParser
        self.parsed = FormulaParser.parse(formula)

    def _denoise_data(self) -> pd.DataFrame:
        """Apply wavelet denoising to the relevant columns."""
        data = self.data.copy()
        dep = self.parsed['dep_var']
        indep = self.parsed['indep_vars']

        vars_to_denoise = []
        if self.denoise_dep:
            vars_to_denoise.append(dep)
        if self.denoise_indep:
            vars_to_denoise.extend(indep)

        # Store wavelet analysis results
        self.wavelet_analyses = {}

        if self.wavelet_method == 'htw':
            wt = HaarATrousWavelet(
                n_levels=self.n_levels,
                threshold_method=self.threshold_method,
            )
            for var in vars_to_denoise:
                if var in data.columns:
                    analysis = wt.full_analysis(data[var].values, name=var)
                    data[var] = analysis['denoised']
                    self.wavelet_analyses[var] = analysis
        else:
            wt = PyWaveletDenoiser(
                wavelet=self.wavelet,
                n_levels=self.n_levels,
                method=self.wavelet_method,
                threshold_method=self.threshold_method,
            )
            for var in vars_to_denoise:
                if var in data.columns:
                    analysis = wt.full_analysis(data[var].values, name=var)
                    data[var] = analysis['denoised']
                    self.wavelet_analyses[var] = analysis

        return data

    def fit(
        self,
        mode: Union[str, Dict, List] = 'grid',
        verbose: bool = False,
        fit_original: bool = True,
    ) -> Dict:
        """
        Estimate the W-NARDL model.

        Parameters
        ----------
        mode : str, dict, or list
            Lag selection mode.
        verbose : bool
        fit_original : bool
            Also fit standard NARDL on original data for comparison.

        Returns
        -------
        dict with keys:
            'wavelet' : NARDLResults on denoised data
            'original' : NARDLResults on original data (if fit_original=True)
            'wavelet_analyses' : dict of wavelet decomposition results
            'comparison' : ComparisonResults
        """
        output = {}

        # 1. Denoise data
        if verbose:
            print("Step 1: Applying wavelet denoising...")
        denoised_data = self._denoise_data()
        output['wavelet_analyses'] = self.wavelet_analyses

        # 2. Fit W-NARDL on denoised data
        if verbose:
            print("\nStep 2: Estimating NARDL on denoised (smoothed) series...")
        wnardl = NARDL(
            data=denoised_data,
            formula=self.formula,
            maxlag=self.maxlag,
            criterion=self.criterion,
            case=self.case,
            different_asym_lag=self.different_asym_lag,
        )
        output['wavelet'] = wnardl.fit(mode=mode, verbose=verbose)

        # 3. Fit standard NARDL on original data
        if fit_original:
            if verbose:
                print("\nStep 3: Estimating NARDL on original series...")
            nardl = NARDL(
                data=self.data,
                formula=self.formula,
                maxlag=self.maxlag,
                criterion=self.criterion,
                case=self.case,
                different_asym_lag=self.different_asym_lag,
            )
            output['original'] = nardl.fit(mode=mode, verbose=verbose)

        # 4. Create comparison
        if fit_original:
            output['comparison'] = WNARDLComparison(
                original=output['original'],
                wavelet=output['wavelet'],
                wavelet_analyses=self.wavelet_analyses,
            )

        return output


class WNARDLComparison:
    """Compare original NARDL and Wavelet NARDL results."""

    def __init__(self, original: NARDLResults, wavelet: NARDLResults,
                 wavelet_analyses: Dict):
        self.original = original
        self.wavelet = wavelet
        self.wavelet_analyses = wavelet_analyses

    def summary(self, print_output: bool = True) -> str:
        """Print comparative summary."""
        o = self.original
        w = self.wavelet

        lines = []
        lines.append("=" * 78)
        lines.append("     WAVELET-NARDL vs STANDARD NARDL — COMPARISON")
        lines.append("=" * 78)
        lines.append("")

        # Model fit comparison
        lines.append(f"  {'Metric':<25s} {'Original':>15s} {'W-NARDL':>15s} {'Better':>10s}")
        lines.append("-" * 78)

        metrics = [
            ('R²', o.rsquared, w.rsquared, 'higher'),
            ('Adj. R²', o.rsquared_adj, w.rsquared_adj, 'higher'),
            ('AIC (normalized)', o.aic_norm, w.aic_norm, 'lower'),
            ('BIC (normalized)', o.bic_norm, w.bic_norm, 'lower'),
            ('HQ (normalized)', o.hq_norm, w.hq_norm, 'lower'),
            ('Log-Likelihood', o.llf, w.llf, 'higher'),
            ('Durbin-Watson', o.durbin_watson, w.durbin_watson, None),
        ]

        for name, orig_val, wave_val, direction in metrics:
            if direction == 'higher':
                better = '◄ W-NARDL' if wave_val > orig_val else '◄ Original'
            elif direction == 'lower':
                better = '◄ W-NARDL' if wave_val < orig_val else '◄ Original'
            else:
                better = ''

            lines.append(f"  {name:<25s} {orig_val:>15.4f} {wave_val:>15.4f} {better:>10s}")

        lines.append("-" * 78)

        # Lag comparison
        lines.append(f"\n  Optimal Lags:")
        lines.append(f"    Original : {o.optimal_lags}")
        lines.append(f"    W-NARDL  : {w.optimal_lags}")

        # Conclusion
        if w.bic_norm < o.bic_norm:
            lines.append(f"\n  ✓ Wavelet denoising improves model fit (lower BIC).")
            lines.append(f"    This confirms the paper's finding that denoising is effective.")
        else:
            lines.append(f"\n  ✗ Standard NARDL has better BIC. Wavelet denoising may not be")
            lines.append(f"    necessary for this pair of variables.")

        lines.append("=" * 78)

        text = "\n".join(lines)
        if print_output:
            print(text)
        return text
