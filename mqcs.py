"""
Panel MQCS Quantile Cointegration Test
Based on: Xiao (2009) and the methodology from the provided paper
Author: Your Name
License: MIT
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats


class MQCSTest:
    """
    Modified Quantile Cointegration Statistic (MQCS) Test
    Tests for cointegration at specified quantiles using residual-based approach
    """
    
    def __init__(self, y, x, tau=0.5, bandwidth=None):
        """
        Parameters:
        -----------
        y : array-like
            Dependent variable (must be 1-D)
        x : array-like
            Independent variable(s) (can be 1-D or 2-D)
        tau : float
            Quantile level (default: 0.5 for median)
        bandwidth : int, optional
            Bandwidth for long-run variance estimation (default: n^(1/3))
        """
        self.y = np.asarray(y).flatten()
        self.x = np.asarray(x)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        
        self.tau = tau
        self.n = len(self.y)
        self.bandwidth = bandwidth if bandwidth is not None else int(self.n**(1/3))
        
        # Validation
        if len(self.y) != len(self.x):
            raise ValueError(f"Length mismatch: y={len(self.y)}, x={len(self.x)}")
        if not np.all(np.isfinite(self.y)) or not np.all(np.isfinite(self.x)):
            raise ValueError("Data contains inf or NaN values")
        if self.n < 20:
            raise ValueError(f"Insufficient observations: n={self.n}, need at least 20")
    
    def estimate_quantile_regression(self):
        """
        Estimate quantile regression: y_t = x'_t * beta(tau) + u_t
        Returns coefficients and residuals
        """
        X = sm.add_constant(self.x)
        model = QuantReg(self.y, X)
        result = model.fit(q=self.tau)
        
        self.beta = result.params
        self.residuals = self.y - X @ self.beta
        
        return self.beta, self.residuals
    
    def compute_psi(self, residuals):
        """
        Compute psi function: psi_tau(u) = tau - I(u < 0)
        """
        return self.tau - (residuals < 0).astype(float)
    
    def compute_long_run_variance(self, psi):
        """
        Compute long-run variance using Newey-West type estimator
        omega_psi = gamma_0 + 2 * sum(gamma_j) for j=1 to h
        """
        h = self.bandwidth
        
        # Compute autocovariances
        gamma_0 = np.var(psi, ddof=1)
        gamma_sum = 0
        
        for lag in range(1, h + 1):
            gamma_j = np.cov(psi[:-lag], psi[lag:])[0, 1]
            gamma_sum += gamma_j
        
        omega = gamma_0 + 2 * gamma_sum
        
        # Ensure positive
        if omega <= 0:
            omega = gamma_0  # Fall back to variance only
        
        return omega
    
    def compute_mqcs_statistic(self):
        """
        Compute MQCS test statistic:
        MQCS(tau) = max |sum(psi_tau(u_t))| / (sqrt(n) * omega_psi)
        """
        # Get residuals
        _, residuals = self.estimate_quantile_regression()
        
        # Compute psi function
        psi = self.compute_psi(residuals)
        
        # Compute long-run variance
        omega = self.compute_long_run_variance(psi)
        
        # Compute cumulative sum process
        cumsum_psi = np.cumsum(psi)
        
        # MQCS statistic: max of absolute cumulative sum, scaled
        mqcs_stat = np.max(np.abs(cumsum_psi)) / (np.sqrt(self.n) * np.sqrt(omega))
        
        return mqcs_stat, psi, omega
    
    def bootstrap_pvalue(self, B=1000, block_size=None, seed=42):
        """
        Compute p-value using moving block bootstrap
        
        Parameters:
        -----------
        B : int
            Number of bootstrap replications
        block_size : int, optional
            Block size for moving block bootstrap (default: n^(1/3))
        seed : int
            Random seed for reproducibility
        """
        if block_size is None:
            block_size = int(self.n**(1/3))
        
        # Compute original statistic
        mqcs_stat, psi, omega = self.compute_mqcs_statistic()
        
        # Bootstrap
        np.random.seed(seed)
        bootstrap_stats = []
        
        for b in range(B):
            # Moving block bootstrap for residuals
            boot_residuals = self._moving_block_bootstrap(self.residuals, block_size)
            
            # Compute psi for bootstrap sample
            boot_psi = self.compute_psi(boot_residuals)
            
            # Compute bootstrap long-run variance
            boot_omega = self.compute_long_run_variance(boot_psi)
            
            # Compute bootstrap statistic
            boot_cumsum = np.cumsum(boot_psi)
            boot_stat = np.max(np.abs(boot_cumsum)) / (np.sqrt(self.n) * np.sqrt(boot_omega))
            
            bootstrap_stats.append(boot_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute p-value: proportion of bootstrap stats >= original stat
        pvalue = 1 - np.mean(bootstrap_stats <= mqcs_stat)
        
        return mqcs_stat, pvalue, bootstrap_stats
    
    def _moving_block_bootstrap(self, data, block_size):
        """
        Moving block bootstrap resampling
        """
        n = len(data)
        n_blocks = int(np.ceil(n / block_size))
        
        boot_sample = []
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            block = data[start_idx:start_idx + block_size]
            boot_sample.extend(block)
        
        return np.array(boot_sample[:n])


def panel_mqcs_test(df, id_col, y_col, x_col, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
                    B=1000, transform='log_shift', normalize=True):
    """
    Run MQCS test on panel data
    
    Parameters:
    -----------
    df : DataFrame
        Panel data
    id_col : str
        Column name for entity ID (e.g., country)
    y_col : str
        Column name for dependent variable
    x_col : str
        Column name for independent variable
    quantiles : list
        List of quantiles to test
    B : int
        Number of bootstrap replications
    transform : str
        Transformation method: 'log_shift', 'ihs', 'log', or 'none'
    normalize : bool
        Whether to apply z-score normalization within entity
    
    Returns:
    --------
    results_df : DataFrame
        Results table with test statistics and p-values for each entity and quantile
    """
    
    results = []
    
    for entity in df[id_col].unique():
        entity_data = df[df[id_col] == entity].copy()
        
        # Transform data (entity-specific)
        y_data = entity_data[y_col].values
        x_data = entity_data[x_col].values
        
        # Apply transformation
        y_data, x_data = _transform_data(y_data, x_data, transform, entity)
        
        # Apply normalization
        if normalize:
            y_data = (y_data - np.mean(y_data)) / np.std(y_data)
            x_data = (x_data - np.mean(x_data)) / np.std(x_data)
        
        # Check validity
        if len(y_data) < 20:
            print(f"Skipping {entity}: insufficient data (n={len(y_data)})")
            continue
        
        if not np.all(np.isfinite(y_data)) or not np.all(np.isfinite(x_data)):
            print(f"Skipping {entity}: contains inf/NaN")
            continue
        
        # Run tests at each quantile
        row = {'Entity': entity, 'N': len(y_data)}
        
        for tau in quantiles:
            try:
                test = MQCSTest(y_data, x_data, tau=tau)
                stat, pval, _ = test.bootstrap_pvalue(B=B)
                
                # Add significance stars
                if pval < 0.01:
                    stars = '***'
                elif pval < 0.05:
                    stars = '**'
                elif pval < 0.10:
                    stars = '*'
                else:
                    stars = ''
                
                row[f'τ={tau:.1f}'] = f"{stat:.3f}{stars}"
                row[f'τ={tau:.1f}_pval'] = pval
                
            except Exception as e:
                print(f"Error for {entity} at τ={tau}: {str(e)}")
                row[f'τ={tau:.1f}'] = 'N/A'
                row[f'τ={tau:.1f}_pval'] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def _transform_data(y, x, method, entity_name):
    """Apply transformation to data"""
    
    if method == 'log_shift':
        # Min-shift for negative values
        y_min = np.min(y)
        x_min = np.min(x)
        
        if y_min <= 0:
            shift_y = abs(y_min) + 1
            y = y + shift_y
            print(f"  {entity_name}: shifted y by +{shift_y:.2f}")
        
        if x_min <= 0:
            shift_x = abs(x_min) + 1
            x = x + shift_x
            print(f"  {entity_name}: shifted x by +{shift_x:.2f}")
        
        y = np.log(y)
        x = np.log(x)
    
    elif method == 'ihs':
        # Inverse hyperbolic sine
        y = np.arcsinh(y)
        x = np.arcsinh(x)
    
    elif method == 'log':
        if np.any(y <= 0) or np.any(x <= 0):
            raise ValueError(f"{entity_name}: has non-positive values, cannot use log")
        y = np.log(y)
        x = np.log(x)
    
    # else: 'none' - no transformation
    
    return y, x


# Critical values from Xiao and Phillips (2002) for constant coefficient case
CRITICAL_VALUES = {
    0.10: 1.616,  # 10% significance
    0.05: 1.842,  # 5% significance
    0.01: 2.326   # 1% significance
}


if __name__ == "__main__":
    # Example usage
    print("MQCS Quantile Cointegration Test - Example")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(123)
    n = 100
    
    # Create panel data
    data = []
    for country in ['China', 'Japan', 'Korea']:
        x = np.cumsum(np.random.randn(n)) + 10
        y = 0.5 * x + np.random.randn(n)
        
        df_country = pd.DataFrame({
            'country': country,
            'y': y,
            'x': x
        })
        data.append(df_country)
    
    df = pd.concat(data, ignore_index=True)
    
    # Run panel MQCS test
    results = panel_mqcs_test(
        df, 
        id_col='country',
        y_col='y',
        x_col='x',
        quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
        B=500,  # Use 1000 for final results
        transform='none',
        normalize=True
    )
    
    print("\nResults:")
    print(results.to_string(index=False))
