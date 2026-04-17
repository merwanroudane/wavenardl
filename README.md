# wavenardl

**Wavelet-based Nonlinear Autoregressive Distributed Lag (W-NARDL) for Python**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The **first Python implementation** of both standard NARDL and Wavelet-based NARDL models. This library provides a complete econometric toolkit for asymmetric cointegration analysis with wavelet denoising.

## Features

### Core Models
- **NARDL** — Nonlinear ARDL with short-run and long-run asymmetry (Shin et al., 2014)
- **Wavelet NARDL** — HTW wavelet-denoised NARDL (Jammazi et al., 2015)
- **ARDL** — Standard linear ARDL as a special case

### Wavelet Methods
- **Haar à Trous Wavelet (HTW)** — Non-decimated transform from the paper
- **PyWavelets Integration** — SWT, DWT with any wavelet family
- **Donoho Thresholding** — Universal soft/hard threshold denoising

### Statistical Tests
- **PSS Bounds F-test** — Pesaran, Shin & Smith (2001) with Cases I–V
- **Narayan Test** — Small-sample critical values (Narayan, 2005)  
- **Symmetry Tests** — Wald tests for long-run and short-run asymmetry
- **Diagnostics** — Breusch-Godfrey, Breusch-Pagan, Jarque-Bera, RESET, CUSUM

### Analysis Tools
- **Dynamic Multipliers** — Cumulative multiplier curves with bootstrap CI
- **Long-run Coefficients** — Delta method standard errors
- **Error Correction Model** — Two-step ECM estimation
- **Automatic Lag Selection** — Grid search, quick search (AIC/BIC/HQ)

### Output & Visualization
- **Publication Tables** — LaTeX, HTML, and rich console output
- **Wavelet Plots** — Decomposition, scalogram heatmaps
- **Multiplier Plots** — Positive/negative effects with CI bands
- **Diagnostic Plots** — Residuals, Q-Q, ACF, CUSUM panels

## Installation

```bash
pip install wavenardl
```

Or from source:
```bash
git clone https://github.com/merwanroudane/wavenardl.git
cd wavenardl
pip install -e .
```

## Quick Start

### Standard NARDL
```python
from wavenardl import NARDL, pss_f_test, symmetry_test

import pandas as pd
data = pd.read_csv('my_data.csv')

# Estimate NARDL with asymmetric exchange rate effects
model = NARDL(
    data=data,
    formula="oil ~ exrate + cpi + Asymmetric(exrate) + deterministic(crisis)",
    maxlag=4,
    criterion='BIC'
)
results = model.fit(mode='grid')
results.summary()

# Cointegration test
pss = pss_f_test(results, case=3)
print(pss)

# Symmetry test
sym = symmetry_test(results)
print(sym)
```

### Wavelet NARDL
```python
from wavenardl import WaveletNARDL

# Fit W-NARDL with HTW denoising (as in the paper)
wmodel = WaveletNARDL(
    data=data,
    formula="oil ~ exrate + Asymmetric(exrate)",
    maxlag=4,
    wavelet_method='htw',  # Haar à Trous from the paper
    n_levels=4
)
results = wmodel.fit(fit_original=True)

# Compare original vs wavelet-denoised NARDL
results['comparison'].summary()
```

### Dynamic Multipliers
```python
from wavenardl import DynamicMultipliers, bootstrap_multipliers, plot_multipliers

# Compute multipliers
dm = DynamicMultipliers(results, horizon=40)
dm.summary()

# Bootstrap confidence intervals
mpsi = bootstrap_multipliers(results, horizon=40, replications=500, confidence_level=95)

# Plot
plot_multipliers(mpsi, variable='exrate', dep_var='oil')
```

### Wavelet Visualization
```python
from wavenardl import HaarATrousWavelet, plot_wavelet_decomposition, plot_scalogram

htw = HaarATrousWavelet(n_levels=6)
analysis = htw.full_analysis(data['oil'].values, name='Oil Price')

# Decomposition plot (paper Fig. 1 style)
plot_wavelet_decomposition(analysis)

# Scalogram heatmap
plot_scalogram(analysis, cmap='magma')
```

### Publication Tables
```python
from wavenardl import summary_table, comparison_table

# LaTeX table for journal submission
latex = summary_table(results, format='latex')
print(latex)

# HTML table for reports
html = summary_table(results, format='html')
```

## Formula Syntax

The formula interface supports R-like specifications:

| Formula Element | Description |
|---|---|
| `y ~ x1 + x2` | Standard ARDL |
| `Asymmetric(x1)` or `asym(x1)` | Both LR and SR asymmetry |
| `Lasymmetric(x1)` | Long-run only asymmetry |
| `Sasymmetric(x1)` | Short-run only asymmetry |
| `deterministic(d1 + d2)` | Exogenous / dummy variables |
| `trend` | Linear time trend |
| `-1` | No intercept |

## Dependencies

- NumPy, Pandas, SciPy, Statsmodels
- PyWavelets (`pywt`)
- Matplotlib
- Rich, Tabulate

## References

- Jammazi, R., Lahiani, A., & Nguyen, D. K. (2015). *A wavelet-based nonlinear ARDL model for assessing the exchange rate pass-through to crude oil prices.* Journal of International Financial Markets, Institutions and Money, 34, 173-187.

- Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014). *Modelling asymmetric cointegration and dynamic multipliers in a nonlinear ARDL framework.* Festschrift in Honor of Peter Schmidt, 281-314.

- Pesaran, M. H., Shin, Y., & Smith, R. (2001). *Bounds testing approaches to the analysis of level relationship.* Journal of Applied Econometrics, 16(3), 289-326.

- Narayan, P. K. (2005). *The saving and investment nexus for China.* Applied Economics, 37(17), 1979-1990.

## Author

**Dr Merwan Roudane**

## License

MIT License
