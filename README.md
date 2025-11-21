# Panel MQCS Quantile Cointegration Test

Implementation of the **Modified Quantile Cointegration Statistic (MQCS)** test for panel data, based on Xiao (2009) methodology.

## Features

- **Entity-specific transformation**: Each panel entity (country, firm, etc.) is transformed separately
- **Multiple transformation methods**: 
  - Min-shift log (handles negative values)
  - Inverse Hyperbolic Sine (IHS)
  - Natural log
  - No transformation
- **Quantile cointegration testing**: Test at any quantile (τ) of the conditional distribution
- **Bootstrap p-values**: Moving block bootstrap for finite-sample inference
- **Panel data support**: Automatic handling of multiple entities

## Installation

### Local Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

### Deploy on Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Select `streamlit_app.py` as main file
5. Deploy!

## Usage

### Streamlit App

```bash
streamlit run streamlit_app.py
```

Then:
1. Upload your panel data (CSV or Excel)
2. Select entity ID column
3. Select dependent (y) and independent (x) variables
4. Choose transformation method
5. Select quantiles to test
6. Run the test!

### Python Script

```python
import pandas as pd
from mqcs_cointegration import panel_mqcs_test

# Load your data
df = pd.read_csv('your_panel_data.csv')

# Run MQCS test
results = panel_mqcs_test(
    df,
    id_col='country',        # Entity identifier
    y_col='gdp',            # Dependent variable
    x_col='investment',     # Independent variable
    quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
    B=1000,                 # Bootstrap replications
    transform='log_shift',  # Transformation method
    normalize=True          # Z-score normalization
)

print(results)
```

### Single Entity Test

```python
from mqcs_cointegration import MQCSTest
import numpy as np

# Your data
y = np.array([...])  # Dependent variable
x = np.array([...])  # Independent variable

# Run test at median (τ=0.5)
test = MQCSTest(y, x, tau=0.5)
stat, pval, bootstrap_stats = test.bootstrap_pvalue(B=1000)

print(f"MQCS statistic: {stat:.3f}")
print(f"P-value: {pval:.3f}")
```

## Data Format

Your panel data should have:

| Entity | Year | Y_variable | X_variable |
|--------|------|------------|------------|
| China  | 2000 | 1000       | 500        |
| China  | 2001 | 1050       | 520        |
| Japan  | 2000 | 800        | 400        |
| Japan  | 2001 | 820        | 410        |

- **Entity column**: Country, firm, individual ID, etc.
- **Y variable**: Dependent variable (e.g., GDP, consumption)
- **X variable**: Independent variable (e.g., investment, income)
- **Optional**: Date/time column for visualization

## Methodology

### The MQCS Test

The MQCS test examines whether there exists a cointegration relationship at a specific quantile τ:

1. **Estimate quantile regression**: 
   ```
   y_t = x'_t * β(τ) + u_t
   ```

2. **Compute residuals**: 
   ```
   û_t = y_t - x'_t * β̂(τ)
   ```

3. **Calculate test statistic**:
   ```
   MQCS(τ) = max|∑ψ_τ(û_t)| / (√n * ω̂)
   ```
   where ψ_τ(u) = τ - I(u < 0)

4. **Bootstrap p-value**: Use moving block bootstrap to account for serial correlation

### Null Hypothesis

H₀: No cointegration at quantile τ

Large values of MQCS(τ) lead to rejection of the null.

### Critical Values

From Xiao & Phillips (2002):
- 10% level: 1.616
- 5% level: 1.842
- 1% level: 2.326

**Note**: Bootstrap p-values are more accurate for finite samples.

## Transformation Methods

### 1. Log with Min-Shift (Recommended for negatives)

```python
# For each entity separately:
if min(x) <= 0:
    x = x + abs(min(x)) + 1
x_transformed = log(x)
```

**Use when**: Data contains negative or zero values

### 2. Inverse Hyperbolic Sine (IHS)

```python
x_transformed = arcsinh(x) = log(x + sqrt(x² + 1))
```

**Use when**: Data contains negatives, approximates log for large values

### 3. Natural Log

```python
x_transformed = log(x)
```

**Use when**: All data is strictly positive

### 4. No Transformation

Use raw data as-is.

## Parameters

### Key Parameters

- `tau`: Quantile level (0 < τ < 1)
  - 0.5 = median regression
  - 0.1, 0.9 = tail behavior
  
- `B`: Bootstrap replications
  - Minimum: 500
  - Recommended: 1000
  - More accurate: 5000 (but slower)

- `transform`: Transformation method
  - 'log_shift': Safest for economic data
  - 'ihs': Modern alternative
  - 'log': Traditional
  - 'none': Raw data

- `normalize`: Within-entity z-score normalization
  - Recommended: True
  - Standardizes each entity separately

## Interpretation

### Test Statistics

```
MQCS(τ=0.5) = 2.936***
```

- **Value**: 2.936 (test statistic)
- **Stars**: *** = p < 0.01 (reject H₀)
- **Interpretation**: Strong evidence of cointegration at median

### Quantile Patterns

Different quantiles reveal different aspects:

- **Lower quantiles (τ=0.1, 0.2)**: Recession/downturn behavior
- **Median (τ=0.5)**: Typical relationship
- **Upper quantiles (τ=0.8, 0.9)**: Boom/expansion behavior

Example from paper:

| Country | τ=0.1 | τ=0.5 | τ=0.9 |
|---------|-------|-------|-------|
| China   | 1.395 | 2.114 | 2.373 |
| Japan   | 1.576 | 2.302 | 4.974*** |

Japan shows strong cointegration in upper quantile (boom times).

## Files

- `mqcs_cointegration.py`: Core implementation
- `streamlit_app.py`: Web interface
- `requirements.txt`: Python dependencies
- `README.md`: This file

## References

1. Xiao, Z. (2009). "Quantile Cointegrating Regression." *Journal of Econometrics*, 150(2), 248-260.

2. Xiao, Z., & Phillips, P. C. (2002). "A CUSUM test for cointegration using regression residuals." *Journal of Econometrics*, 108(1), 43-61.

3. Moving block bootstrap: Politis, D. N., & Romano, J. P. (1994). "The stationary bootstrap." *Journal of the American Statistical Association*, 89(428), 1303-1313.

## License

MIT License

## Contributing

Issues and pull requests welcome!

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mqcs_panel_test,
  author = {Your Name},
  title = {Panel MQCS Quantile Cointegration Test},
  year = {2024},
  url = {https://github.com/yourusername/mqcs-test}
}
```

## Support

For issues or questions:
- Open an issue on GitHub
- Email: your.email@example.com
