"""
Streamlit App: Panel MQCS Quantile Cointegration Test
Based on Xiao (2009) methodology
Combined version with MQCS implementation included
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# =============================================================================
# MQCS COINTEGRATION IMPLEMENTATION (formerly in mcqs.py)
# =============================================================================

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
            if len(psi[:-lag]) > 0 and len(psi[lag:]) > 0:
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
        pvalue = np.mean(bootstrap_stats >= mqcs_stat)
        
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

def _transform_data(y, x, method, entity_name):
    """Apply transformation to data"""
    
    if method == 'log_shift':
        # Min-shift for negative values
        y_min = np.min(y)
        x_min = np.min(x)
        
        if y_min <= 0:
            shift_y = abs(y_min) + 1
            y = y + shift_y
        
        if x_min <= 0:
            shift_x = abs(x_min) + 1
            x = x + shift_x
        
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
        try:
            y_data, x_data = _transform_data(y_data, x_data, transform, entity)
        except Exception as e:
            st.warning(f"Skipping {entity}: transformation error - {str(e)}")
            continue
        
        # Apply normalization
        if normalize:
            y_data = (y_data - np.mean(y_data)) / np.std(y_data)
            x_data = (x_data - np.mean(x_data)) / np.std(x_data)
        
        # Check validity
        if len(y_data) < 20:
            st.warning(f"Skipping {entity}: insufficient data (n={len(y_data)})")
            continue
        
        if not np.all(np.isfinite(y_data)) or not np.all(np.isfinite(x_data)):
            st.warning(f"Skipping {entity}: contains inf/NaN")
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
                st.warning(f"Error for {entity} at τ={tau}: {str(e)}")
                row[f'τ={tau:.1f}'] = 'N/A'
                row[f'τ={tau:.1f}_pval'] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)

# Critical values from Xiao and Phillips (2002) for constant coefficient case
CRITICAL_VALUES = {
    0.10: 1.616,  # 10% significance
    0.05: 1.842,  # 5% significance
    0.01: 2.326   # 1% significance
}

# =============================================================================
# STREAMLIT APP (your original coint.py)
# =============================================================================

st.set_page_config(page_title="MQCS Quantile Cointegration", layout="wide")

st.title("📊 Panel MQCS Quantile Cointegration Test")
st.markdown("""
This app implements the **Modified Quantile Cointegration Statistic (MQCS)** test for panel data.
The test examines cointegration relationships at different quantiles of the conditional distribution.
""")

# File upload
uploaded = st.file_uploader("Upload panel CSV or Excel file", type=["csv", "xlsx"])

if uploaded is None:
    st.info("👆 Please upload your panel data file to begin")
    st.markdown("""
    **Expected data format:**
    - Panel data with entity identifier (e.g., Country, ID)
    - Time series for each entity
    - At least one dependent variable (y)
    - At least one independent variable (x)
    """)
    st.stop()

# Load data
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

df_raw = load_data(uploaded)

st.write("### 📁 Raw Data Preview")
st.dataframe(df_raw.head(10))
st.write(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# Column selection
st.write("### ⚙️ Configure Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    id_col = st.selectbox("Entity ID Column (e.g., Country)", df_raw.columns)

with col2:
    y_col = st.selectbox("Dependent Variable (y)", 
                         [c for c in df_raw.columns if c != id_col])

with col3:
    x_col = st.selectbox("Independent Variable (x)", 
                         [c for c in df_raw.columns if c not in [id_col, y_col]])

# Transformation settings
st.write("### 🔧 Transformation Settings")

col1, col2, col3 = st.columns(3)

with col1:
    transform = st.selectbox(
        "Transformation Method",
        ["log_shift", "ihs", "log", "none"],
        help="""
        - log_shift: Min-shift then log (handles negatives)
        - ihs: Inverse Hyperbolic Sine (handles negatives)
        - log: Natural log (requires positive values)
        - none: No transformation
        """
    )

with col2:
    normalize = st.checkbox("Z-score normalization (within entity)", value=True,
                           help="Standardize each entity to mean=0, std=1")

with col3:
    B = st.number_input("Bootstrap replications", min_value=100, max_value=5000, 
                       value=1000, step=100,
                       help="More replications = more accurate p-values but slower")

# Quantile selection
st.write("### 📈 Quantile Selection")
use_preset = st.radio("Quantile selection:", ["Preset quantiles", "Custom quantiles"])

if use_preset == "Preset quantiles":
    quantile_option = st.selectbox(
        "Select preset",
        ["Standard (0.1, 0.3, 0.5, 0.7, 0.9)",
         "Fine (0.1 to 0.9 by 0.1)",
         "Tails focus (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)"]
    )
    
    if quantile_option.startswith("Standard"):
        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    elif quantile_option.startswith("Fine"):
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        quantiles = [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]
else:
    quantile_input = st.text_input("Enter quantiles (comma-separated)", "0.1, 0.3, 0.5, 0.7, 0.9")
    try:
        quantiles = [float(q.strip()) for q in quantile_input.split(",")]
        if not all(0 < q < 1 for q in quantiles):
            st.error("All quantiles must be between 0 and 1")
            st.stop()
    except:
        st.error("Invalid quantile format. Use comma-separated decimals (e.g., 0.1, 0.5, 0.9)")
        st.stop()

st.write(f"**Testing at quantiles:** {quantiles}")

# Run test button
if st.button("🚀 Run MQCS Test", type="primary"):
    
    with st.spinner("Running MQCS test... This may take a few minutes..."):
        
        try:
            # Prepare data
            df_test = df_raw[[id_col, y_col, x_col]].dropna()
            
            st.write(f"**Entities:** {df_test[id_col].nunique()}")
            st.write(f"**Total observations:** {len(df_test)}")
            
            # Run test
            results = panel_mqcs_test(
                df_test,
                id_col=id_col,
                y_col=y_col,
                x_col=x_col,
                quantiles=quantiles,
                B=B,
                transform=transform,
                normalize=normalize
            )
            
            if len(results) == 0:
                st.error("❌ No valid results. Check warnings above.")
                st.stop()
            
            # Display results
            st.write("## 📊 Results")
            st.success(f"✅ Successfully tested {len(results)} entities")
            
            # Prepare display table (without p-values for cleaner view)
            display_cols = ['Entity', 'N'] + [col for col in results.columns if not col.endswith('_pval') and col not in ['Entity', 'N']]
            display_df = results[display_cols].copy()
            
            st.dataframe(display_df, use_container_width=True)
            
            # Interpretation guide
            st.write("### 📖 Interpretation Guide")
            st.markdown(f"""
            **Test Statistic Interpretation:**
            - Larger values indicate **rejection** of the null hypothesis (no cointegration)
            - Stars indicate significance: *** p<0.01, ** p<0.05, * p<0.10
            
            **Critical Values (Xiao & Phillips 2002):**
            - 10% level: {CRITICAL_VALUES[0.10]}
            - 5% level: {CRITICAL_VALUES[0.05]}
            - 1% level: {CRITICAL_VALUES[0.01]}
            
            **Note:** P-values are computed via {B} bootstrap replications.
            Bootstrap accounts for finite-sample properties and serial correlation.
            """)
            
            # Full results with p-values
            with st.expander("📋 Full Results (including p-values)"):
                st.dataframe(results, use_container_width=True)
            
            # Download buttons
            st.write("### 💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "mqcs_results.csv",
                    "text/csv"
                )
            
            with col2:
                csv_full = results.to_csv(index=False)
                st.download_button(
                    "📥 Download Full Results with P-values (CSV)",
                    csv_full,
                    "mqcs_results_full.csv",
                    "text/csv"
                )
            
            # Summary statistics
            st.write("### 📈 Summary Statistics")
            
            # Count significant results
            sig_counts = {}
            for tau in quantiles:
                col_name = f'τ={tau:.1f}_pval'
                if col_name in results.columns:
                    sig_counts[f'τ={tau:.1f}'] = {
                        '1%': (results[col_name] < 0.01).sum(),
                        '5%': (results[col_name] < 0.05).sum(),
                        '10%': (results[col_name] < 0.10).sum()
                    }
            
            sig_df = pd.DataFrame(sig_counts).T
            sig_df.columns = ['Sig at 1%', 'Sig at 5%', 'Sig at 10%']
            
            st.write("**Number of entities with significant cointegration:**")
            st.dataframe(sig_df)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

# Footer
st.write("---")
st.markdown("""
**Reference:**
- Xiao, Z. (2009). "Quantile Cointegrating Regression." *Journal of Econometrics*.
- Test implements moving block bootstrap for p-value computation
- Entity-specific transformation ensures valid panel cointegration tests
""")
