# Wavelet NARDL Python Library — Implementation Plan

## Goal

Build a **complete Python library** called `wavenardl` that converts two R packages into Python:

1. **kardl** (R) → Standard NARDL with full diagnostics, tests, multipliers, bootstrap
2. **DWaveNARDL** (R) → Wavelet-based NARDL using HTW denoising (Jammazi et al., 2015)

**Neither NARDL nor Wavelet-NARDL exist as Python packages**, making this a first-of-its-kind contribution.

---

## User Review Required

> [!IMPORTANT]
> This is a large library with ~15 Python modules. The full build will take significant time. Please confirm:
> 1. Package name: `wavenardl` — acceptable?
> 2. Should we include example datasets (e.g., oil prices + exchange rates from the paper)?
> 3. Target Python version: 3.9+?

> [!WARNING]
> The R `kardl` package uses `nlWaldTest` and `car::linearHypothesis` for Wald tests. Python equivalents exist in `statsmodels` but may produce slightly different p-values due to numerical differences. We will validate against R outputs.

---

## Proposed Changes

### 1. Package Structure

```
wavenardl/
├── wavenardl/
│   ├── __init__.py              # Public API exports
│   ├── nardl.py                 # Core NARDL/ARDL estimation class
│   ├── wavelet.py               # HTW wavelet denoising (Jammazi 2015)
│   ├── wavenardl.py             # Combined W-NARDL model
│   ├── prepare.py               # Data preparation, asymmetric decomposition
│   ├── lagselect.py             # Automatic lag selection (AIC/BIC/HQ/AICc)
│   ├── tests.py                 # PSS F-test, PSS t-test, Narayan, symmetry tests
│   ├── multipliers.py           # Dynamic multipliers + bootstrap CI
│   ├── longrun.py               # Long-run coefficients via delta method
│   ├── ecm.py                   # Error Correction Model representation
│   ├── diagnostics.py           # Serial correlation, heteroskedasticity, normality, CUSUM
│   ├── visualize.py             # Beautiful plots: multipliers, wavelets, diagnostics
│   ├── tables.py                # Publication-quality tables (LaTeX, HTML, console)
│   ├── critical_values.py       # PSS & Narayan critical value tables
│   └── utils.py                 # Shared utilities
├── examples/
│   ├── example_nardl.py
│   ├── example_wavelet_nardl.py
│   └── data/                    # Sample datasets
├── tests/
│   └── test_nardl.py
├── docs/                        # Documentation website
├── setup.py
├── pyproject.toml
└── README.md
```

---

### 2. Core Modules

---

#### [NEW] `wavenardl/prepare.py` — Data Preparation & Asymmetric Decomposition

Converts from R's `prepare.R` and `CreateNewVars()`:

- **Partial sum decomposition**: Split variable X into X⁺ (cumulative positive changes) and X⁻ (cumulative negative changes) per Shin et al. (2013)
- **Differencing**: Create ΔY, ΔX⁺, ΔX⁻ and their lags
- **Level lags**: Create Y_{t-1}, X⁺_{t-1}, X⁻_{t-1} for long-run part
- **Formula parsing**: Support flexible specification like `y ~ x + Asymmetric(x) + deterministic(dummy) + trend`
- **Deterministic variables**: Intercept, trend, dummies

---

#### [NEW] `wavenardl/nardl.py` — Core NARDL Estimation

The main model class converting from R's `kardl.R`:

```python
class NARDL:
    def __init__(self, data, formula, maxlag=4, criterion='BIC',
                 case=3, different_asym_lag=False):
        """
        Nonlinear ARDL model (Shin et al., 2013).
        
        Parameters
        ----------
        data : pd.DataFrame
        formula : str  
            e.g. "y ~ x1 + x2 + Asymmetric(x1) + trend"
        maxlag : int
        criterion : str, one of 'AIC', 'BIC', 'AICc', 'HQ'
        case : int, 1-5 (PSS case for deterministics)
        different_asym_lag : bool
        """
    
    def fit(self, mode='grid'):
        """Estimate with automatic lag selection."""
    
    def summary(self):
        """Rich summary table with coefficients, diagnostics."""
    
    def predict(self, newdata=None):
        """Fitted values / out-of-sample prediction."""
```

**Key features from kardl:**
- Grid search, quick search, and user-defined lag modes
- Multiple information criteria (AIC, BIC, AICc, HQ) with normalization
- Both short-run and long-run asymmetric decomposition options
- Flexible formula interface supporting `Asymmetric()`, `Sasymmetric()`, `Lasymmetric()`, `deterministic()`

---

#### [NEW] `wavenardl/wavelet.py` — Wavelet Denoising

Implements the **Haar à Trous Wavelet (HTW)** from the paper (Section 2.1):

```python
class HaarATrousWavelet:
    """
    Implements the non-decimated Haar à Trous wavelet transform 
    for denoising financial time series.
    
    Based on Jammazi et al. (2015) and Murtagh et al. (2004).
    """
    
    def decompose(self, signal, n_levels=None):
        """
        Decompose signal into detail coefficients d_j and smooth s_J.
        
        Uses the non-symmetric filter h = (1/2, 1/2) per Eq. (8)-(9).
        """
    
    def denoise(self, signal, threshold='donoho'):
        """
        Denoise using soft/hard thresholding.
        
        Threshold: λ = sqrt(2σ²log(N)) per Donoho (1995).
        σ estimated from first-level detail coefficients.
        """
    
    def reconstruct(self, coefficients):
        """Reconstruct signal from wavelet coefficients: x(t) = s_J + Σd_j"""
```

Also supports **MODWT** (Maximum Overlap DWT) using `pywt` library for comparison.

**Visualization features:**
- Scalogram with beautiful color maps (viridis, plasma, magma)
- Original vs denoised overlay plots  
- Detail coefficient plots at each scale
- 3D wavelet surface plots

---

#### [NEW] `wavenardl/wavenardl.py` — Combined Wavelet + NARDL

```python
class WaveletNARDL:
    """
    Wavelet-based Nonlinear ARDL model (Jammazi et al., 2015).
    
    Pipeline:
    1. Apply HTW wavelet denoising to dependent & independent variables
    2. Estimate NARDL on denoised series
    3. Compare with standard NARDL on original series
    """
    
    def fit(self, denoise_y=True, denoise_x=True, 
            wavelet='haar', n_levels=None, threshold='donoho'):
        """Fit W-NARDL model."""
    
    def compare(self):
        """Compare denoised vs original NARDL (SIC comparison as in paper)."""
```

---

#### [NEW] `wavenardl/lagselect.py` — Lag Selection

From R's grid/quick/user modes:

- **Grid search**: Exhaustive search over all lag combinations up to maxlag
- **Quick search**: Sequential search (faster)  
- **User-defined**: Direct specification of lag vector
- **Criteria**: AIC, BIC, AICc, HQ (normalized by n, matching kardl)

---

#### [NEW] `wavenardl/tests.py` — Cointegration & Symmetry Tests

From R's `tests.R`:

| Test | R Source | Description |
|------|----------|-------------|
| `pss_f_test()` | `pssf()` | PSS Bounds F-test (Pesaran et al., 2001) with full critical value tables for Cases I-V |
| `pss_t_test()` | `psst()` | PSS t-test for ECM models |
| `narayan_test()` | `narayan()` | Small-sample F-test (Narayan, 2005) with critical values for n=30-80 |
| `symmetry_test()` | `symmetrytest()` | Wald tests for short-run and long-run symmetry |

Critical value tables embedded from: Pesaran, Shin & Smith (2001) Tables CI-CV, and Narayan (2005).

---

#### [NEW] `wavenardl/multipliers.py` — Dynamic Multipliers & Bootstrap

From R's `multipliers.R`:

```python
class DynamicMultipliers:
    """
    Compute cumulative dynamic multipliers ψ_h⁺ and ψ_h⁻.
    
    ψ_h⁺ = Σ(j=0 to h) ∂y_{t+j}/∂x_t⁺
    ψ_h⁻ = Σ(j=0 to h) ∂y_{t+j}/∂x_t⁻
    """
    
    def compute(self, horizon=80, min_pvalue=0):
        """Compute multipliers using omega recursion."""
    
    def bootstrap(self, replications=100, level=95):
        """Bootstrap confidence intervals for asymmetry curve."""
    
    def plot(self, variable=None):
        """Beautiful multiplier plots with CI bands."""
```

---

#### [NEW] `wavenardl/longrun.py` — Long-Run Multipliers

From R's `longrun.R`:

- Compute: LR_i = -β_i / β_dep  
- Standard errors via **delta method**: SE² = A²·V(β_i) + 2AB·Cov(β_i,β_dep) + B²·V(β_dep)
- t-statistics and p-values

---

#### [NEW] `wavenardl/ecm.py` — Error Correction Model

From R's `ecm()` function in kardl:

- Two-step ECM estimation (Cases 1-5)
- Long-run equation → residuals → lagged ECM term
- Short-run equation with ECM term
- Speed of adjustment coefficient validation

---

#### [NEW] `wavenardl/diagnostics.py` — Diagnostic Tests

Additional tests not in the R packages but valuable:

| Test | Purpose |
|------|---------|
| Breusch-Godfrey | Serial correlation |
| Breusch-Pagan | Heteroskedasticity |
| Jarque-Bera | Normality of residuals |
| Ramsey RESET | Functional form |
| CUSUM / CUSUMSQ | Parameter stability |
| Durbin-Watson | First-order autocorrelation |

---

#### [NEW] `wavenardl/visualize.py` — Beautiful Visualizations

Premium visualization module with consistent dark/light themes:

| Plot | Description | Colors |
|------|-------------|--------|
| `plot_wavelet_decomposition()` | Original vs denoised series (as in paper Fig. 1) | Blue (smoothed) / Red (original) |
| `plot_scalogram()` | Wavelet power spectrum heatmap | Viridis/Plasma/Magma colormap |
| `plot_multipliers()` | Dynamic multipliers with bootstrap CI | Blue (positive) / Red (negative) / Gray CI |
| `plot_cusum()` | CUSUM stability test | Green bands |
| `plot_residuals()` | Residual diagnostics panel (4 plots) | Modern palette |
| `plot_coefficients()` | Coefficient comparison bar chart | Gradient colors |
| `plot_lag_selection()` | IC values across lag configurations | Multi-line with markers |
| `plot_asymmetry_heatmap()` | Heatmap of asymmetric effects across variables | Custom diverging colormap |
| `plot_comparison()` | Side-by-side original vs wavelet NARDL | Split panel |

All plots use:
- **Google Font**: Inter or Roboto via matplotlib
- **Dark mode** option
- **Glassmorphism-inspired** transparent overlays for CI bands
- **Smooth gradients** for heatmaps
- **Export**: PNG, SVG, PDF at publication quality (300 DPI)

---

#### [NEW] `wavenardl/tables.py` — Publication-Quality Tables

| Table | Format |
|-------|--------|
| `summary_table()` | Rich console output with stars (*, **, ***) |
| `to_latex()` | LaTeX table ready for journals |
| `to_html()` | Styled HTML for reports |
| `comparison_table()` | Original vs Wavelet NARDL side-by-side |
| `diagnostics_table()` | All diagnostic test results |
| `bounds_test_table()` | PSS/Narayan results with critical values |

---

#### [NEW] `wavenardl/critical_values.py` — Embedded Critical Values

Full tables from:
- Pesaran, Shin & Smith (2001): Cases I-V, k=0-10, significance 1%, 2.5%, 5%, 10%
- Narayan (2005): Cases II-V, k=0-7, n=30-80, significance 1%, 5%, 10%

---

### 3. Package Configuration

#### [NEW] `pyproject.toml`

```toml
[project]
name = "wavenardl"
version = "1.0.0"
description = "Wavelet-based Nonlinear ARDL (W-NARDL) model for Python"
dependencies = [
    "numpy>=1.21",
    "pandas>=1.3",
    "scipy>=1.7",
    "statsmodels>=0.13",
    "matplotlib>=3.5",
    "pywt>=1.1",           # PyWavelets for wavelet transforms
    "tabulate>=0.8",       # Pretty tables
    "rich>=12.0",          # Beautiful console output
]
```

---

## Key Design Decisions

1. **Wavelet library**: Use `pywt` (PyWavelets) for standard wavelet transforms, but implement the **Haar à Trous** (non-decimated) manually since `pywt` doesn't support the exact HTW algorithm from the paper with the `h=(1/2, 1/2)` filter.

2. **OLS engine**: Use `statsmodels.OLS` for regression — gives us coefficient inference, Wald tests, and diagnostic tests out of the box.

3. **Formula parsing**: Build a lightweight parser that converts our R-like formula syntax to statsmodels-compatible form while extracting asymmetric/deterministic/trend specifications.

4. **Partial sum decomposition**: Implement exactly as in the R code: cumulative sum of max(ΔX, 0) for X⁺ and cumulative sum of min(ΔX, 0) for X⁻.

5. **Critical values**: Embed all tables directly (not download) for offline use, matching the exact tables in the R code.

---

## Verification Plan

### Automated Tests
1. **Unit tests**: Verify partial sum decomposition matches R output
2. **Integration test**: Run full NARDL on sample data, compare coefficients with R's kardl output
3. **Wavelet test**: Verify HTW denoising on known signal matches R's `modwt()` output
4. **Bounds test**: Compare PSS F-statistics and decisions with R
5. **Multiplier test**: Verify dynamic multiplier computation against R

### Manual Verification
- Run example scripts and inspect output tables and plots
- Compare BIC/AIC values with R package outputs
- Visual inspection of wavelet decomposition plots
