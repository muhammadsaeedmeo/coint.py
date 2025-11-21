"""
Streamlit App: Panel MQCS Quantile Cointegration & Quantile Granger Causality Tests
Based on Xiao (2009) and Troster (2018) methodologies
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# =============================================================================
# MQCS COINTEGRATION IMPLEMENTATION
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

# =============================================================================
# QUANTILE GRANGER CAUSALITY IMPLEMENTATION
# =============================================================================

class QuantileGrangerCausality:
    """
    Quantile Granger Causality Test based on Troster (2018)
    Tests for Granger causality across different quantiles
    """
    
    def __init__(self, x, y, max_lag=4, quantiles=None):
        """
        Parameters:
        -----------
        x : array-like
            Potential causing variable
        y : array-like
            Response variable
        max_lag : int
            Maximum lag length to test
        quantiles : list
            Quantile levels to test (default: [0.1, 0.3, 0.5, 0.7, 0.9])
        """
        self.x = np.asarray(x).flatten()
        self.y = np.asarray(y).flatten()
        self.max_lag = max_lag
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.3, 0.5, 0.7, 0.9]
        self.n = len(self.x)
        
        # Validation
        if len(self.x) != len(self.y):
            raise ValueError(f"Length mismatch: x={len(self.x)}, y={len(self.y)}")
        if self.n < 30:
            raise ValueError(f"Insufficient observations: n={self.n}, need at least 30")
    
    def prepare_lagged_data(self, lag):
        """
        Prepare lagged data for quantile regression
        """
        n_obs = self.n - lag
        X = np.ones((n_obs, 2 * lag + 1))  # +1 for constant
        
        for i in range(lag):
            X[:, i] = self.y[lag - i - 1:self.n - i - 1]  # Lags of y
            X[:, lag + i] = self.x[lag - i - 1:self.n - i - 1]  # Lags of x
        
        y_current = self.y[lag:]
        
        return X, y_current
    
    def quantile_granger_test(self, lag, tau):
        """
        Test Granger causality at specific quantile and lag
        """
        X, y = self.prepare_lagged_data(lag)
        
        # Restricted model (only lags of y)
        X_restricted = X[:, :lag + 1]  # Lags of y + constant
        
        # Unrestricted model (lags of y and x)
        X_unrestricted = X
        
        # Fit quantile regressions
        try:
            # Restricted model
            model_restricted = QuantReg(y, X_restricted)
            result_restricted = model_restricted.fit(q=tau)
            resid_restricted = result_restricted.resid
            loss_restricted = np.sum(np.abs(resid_restricted * (tau - (resid_restricted < 0))))
            
            # Unrestricted model
            model_unrestricted = QuantReg(y, X_unrestricted)
            result_unrestricted = model_unrestricted.fit(q=tau)
            resid_unrestricted = result_unrestricted.resid
            loss_unrestricted = np.sum(np.abs(resid_unrestricted * (tau - (resid_unrestricted < 0))))
            
            # Test statistic (quantile version of F-test)
            if loss_restricted > 0:
                qgranger_stat = (loss_restricted - loss_unrestricted) / loss_restricted
            else:
                qgranger_stat = 0
            
            return qgranger_stat, result_unrestricted, result_restricted
            
        except Exception as e:
            st.warning(f"Error in quantile regression at τ={tau}, lag={lag}: {str(e)}")
            return np.nan, None, None
    
    def bootstrap_pvalue(self, lag, tau, B=500, seed=42):
        """
        Bootstrap p-value for quantile Granger causality test
        """
        original_stat, _, _ = self.quantile_granger_test(lag, tau)
        
        if np.isnan(original_stat):
            return np.nan, np.nan
        
        # Bootstrap under null hypothesis (no causality)
        np.random.seed(seed)
        bootstrap_stats = []
        
        for b in range(B):
            # Generate bootstrap sample under null (shuffle x)
            x_shuffled = np.random.permutation(self.x)
            
            # Create bootstrap test object
            bootstrap_test = QuantileGrangerCausality(x_shuffled, self.y, self.max_lag, [tau])
            
            # Compute bootstrap statistic
            boot_stat, _, _ = bootstrap_test.quantile_granger_test(lag, tau)
            
            if not np.isnan(boot_stat):
                bootstrap_stats.append(boot_stat)
        
        if len(bootstrap_stats) == 0:
            return original_stat, np.nan
        
        bootstrap_stats = np.array(bootstrap_stats)
        pvalue = np.mean(bootstrap_stats >= original_stat)
        
        return original_stat, pvalue
    
    def run_all_tests(self, B=500):
        """
        Run quantile Granger causality tests for all lags and quantiles
        """
        results = []
        
        for lag in range(1, self.max_lag + 1):
            for tau in self.quantiles:
                try:
                    stat, pval = self.bootstrap_pvalue(lag, tau, B)
                    
                    # Add significance stars
                    if pval < 0.01:
                        stars = '***'
                    elif pval < 0.05:
                        stars = '**'
                    elif pval < 0.10:
                        stars = '*'
                    else:
                        stars = ''
                    
                    results.append({
                        'Lag': lag,
                        'Quantile': tau,
                        'Statistic': round(stat, 4) if not np.isnan(stat) else 'N/A',
                        'P-value': f"{pval:.4f}{stars}" if not np.isnan(pval) else 'N/A',
                        'Raw_Stat': stat,
                        'Raw_Pval': pval
                    })
                    
                except Exception as e:
                    st.warning(f"Error at lag={lag}, τ={tau}: {str(e)}")
                    results.append({
                        'Lag': lag,
                        'Quantile': tau,
                        'Statistic': 'N/A',
                        'P-value': 'N/A',
                        'Raw_Stat': np.nan,
                        'Raw_Pval': np.nan
                    })
        
        return pd.DataFrame(results)

def panel_quantile_granger_causality(df, id_col, x_col, y_col, max_lag=4, 
                                   quantiles=None, B=500, transform='none', normalize=True):
    """
    Run quantile Granger causality test on panel data
    """
    if quantiles is None:
        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = []
    
    for entity in df[id_col].unique():
        entity_data = df[df[id_col] == entity].copy()
        
        if len(entity_data) < 30:
            st.warning(f"Skipping {entity}: insufficient data (n={len(entity_data)})")
            continue
        
        # Extract variables
        x_data = entity_data[x_col].values
        y_data = entity_data[y_col].values
        
        # Apply transformations
        try:
            x_data, y_data = _transform_data(x_data, y_data, transform, entity)
        except Exception as e:
            st.warning(f"Skipping {entity}: transformation error - {str(e)}")
            continue
        
        # Apply normalization
        if normalize:
            x_data = (x_data - np.mean(x_data)) / np.std(x_data)
            y_data = (y_data - np.mean(y_data)) / np.std(y_data)
        
        # Check validity
        if not np.all(np.isfinite(x_data)) or not np.all(np.isfinite(y_data)):
            st.warning(f"Skipping {entity}: contains inf/NaN")
            continue
        
        # Run quantile Granger causality test
        try:
            qgc_test = QuantileGrangerCausality(x_data, y_data, max_lag, quantiles)
            entity_results = qgc_test.run_all_tests(B)
            
            # Add entity identifier
            entity_results['Entity'] = entity
            entity_results['N'] = len(entity_data)
            
            results.append(entity_results)
            
        except Exception as e:
            st.warning(f"Error running quantile Granger causality for {entity}: {str(e)}")
    
    if len(results) == 0:
        return pd.DataFrame()
    
    return pd.concat(results, ignore_index=True)

def traditional_granger_causality(df, id_col, x_col, y_col, max_lag=4):
    """
    Run traditional Granger causality tests for comparison
    """
    results = []
    
    for entity in df[id_col].unique():
        entity_data = df[df[id_col] == entity].copy()
        
        if len(entity_data) < 30:
            continue
        
        x_data = entity_data[x_col].values
        y_data = entity_data[y_col].values
        
        # Combine into matrix for VAR
        data = np.column_stack([y_data, x_data])
        
        for lag in range(1, max_lag + 1):
            try:
                # Granger causality test
                gc_result = grangercausalitytests(data, maxlag=lag, verbose=False)
                
                # Get F-test results (most common)
                f_stat = gc_result[lag][0]['ssr_ftest'][0]
                f_pval = gc_result[lag][0]['ssr_ftest'][1]
                
                # Add significance stars
                if f_pval < 0.01:
                    stars = '***'
                elif f_pval < 0.05:
                    stars = '**'
                elif f_pval < 0.10:
                    stars = '*'
                else:
                    stars = ''
                
                results.append({
                    'Entity': entity,
                    'Lag': lag,
                    'Test': 'Traditional Granger',
                    'Statistic': round(f_stat, 4),
                    'P-value': f"{f_pval:.4f}{stars}",
                    'Raw_Stat': f_stat,
                    'Raw_Pval': f_pval
                })
                
            except Exception as e:
                st.warning(f"Error in traditional Granger test for {entity} at lag {lag}: {str(e)}")
    
    return pd.DataFrame(results)

# =============================================================================
# COMMON UTILITY FUNCTIONS
# =============================================================================

def _transform_data(y, x, method, entity_name=""):
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

# [REST OF THE COINTEGRATION FUNCTIONS REMAIN THE SAME - pedroni_panel_cointegration, 
# kao_panel_cointegration, fisher_combined_cointegration, panel_mqcs_test, 
# panel_wide_mqcs_test, run_panel_cointegration_tests]

# =============================================================================
# STREAMLIT APP
# =============================================================================

st.set_page_config(page_title="Panel Quantile Analysis", layout="wide")

st.title("📊 Panel Quantile Cointegration & Granger Causality Tests")
st.markdown("""
This app implements advanced panel data analysis including:
- **Quantile Cointegration Tests** (Xiao, 2009)
- **Quantile Granger Causality Tests** (Troster, 2018)  
- **Traditional Panel Cointegration Tests**
- **Traditional Granger Causality Tests**
""")

# File upload
uploaded = st.file_uploader("Upload panel CSV or Excel file", type=["csv", "xlsx"])

if uploaded is None:
    st.info("👆 Please upload your panel data file to begin")
    st.markdown("""
    **Expected data format:**
    - Panel data with entity identifier (e.g., Country, ID)
    - Time series for each entity
    - At least two variables for analysis
    """)
    st.stop()

# Load data function
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file, engine='openpyxl')
    except:
        try:
            return pd.read_excel(file, engine='xlrd')
        except:
            st.error("Cannot read Excel file. Please ensure openpyxl is installed.")
            st.stop()

df_raw = load_data(uploaded)

st.write("### 📁 Raw Data Preview")
st.dataframe(df_raw.head(10))
st.write(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# Analysis type selection
st.write("### 🔄 Analysis Type")
analysis_type = st.radio(
    "Select analysis type:",
    [
        "Quantile Cointegration Tests", 
        "Quantile Granger Causality Tests",
        "Traditional Panel Cointegration Tests",
        "Traditional Granger Causality Tests"
    ],
    help="""
    - Quantile Cointegration: Test long-run relationships at different quantiles
    - Quantile Granger Causality: Test causal relationships at different quantiles
    - Traditional Panel Cointegration: Standard panel cointegration tests
    - Traditional Granger Causality: Standard Granger causality tests
    """
)

# Column selection
st.write("### ⚙️ Configure Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    # Auto-detect entity column
    possible_entity_cols = [col for col in df_raw.columns if df_raw[col].dtype == 'object' and df_raw[col].nunique() < 100]
    id_col = st.selectbox("Entity ID Column", possible_entity_cols if possible_entity_cols else df_raw.columns)

with col2:
    if "Causality" in analysis_type:
        cause_var = st.selectbox("Potential Causing Variable (X)", 
                                [c for c in df_raw.columns if c != id_col])
    else:
        y_col = st.selectbox("Dependent Variable (Y)", 
                            [c for c in df_raw.columns if c != id_col])

with col3:
    if "Causality" in analysis_type:
        effect_var = st.selectbox("Effect Variable (Y)", 
                                 [c for c in df_raw.columns if c not in [id_col, cause_var]])
    else:
        x_col = st.selectbox("Independent Variable (X)", 
                            [c for c in df_raw.columns if c not in [id_col, y_col]])

# Common settings
st.write("### 🔧 Common Settings")

col1, col2 = st.columns(2)

with col1:
    if "Quantile" in analysis_type:
        use_preset = st.radio("Quantile selection:", ["Preset quantiles", "Custom quantiles"])
        
        if use_preset == "Preset quantiles":
            quantile_option = st.selectbox(
                "Select quantile preset",
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

with col2:
    if "Quantile" in analysis_type or "Causality" in analysis_type:
        max_lag = st.number_input("Maximum Lag Length", min_value=1, max_value=10, value=4)
    
    if "Quantile" in analysis_type:
        B = st.number_input("Bootstrap Replications", min_value=100, max_value=2000, value=500)
    
    if analysis_type in ["Quantile Cointegration Tests", "Quantile Granger Causality Tests"]:
        transform = st.selectbox(
            "Transformation Method",
            ["none", "log_shift", "ihs", "log"],
            help="Data transformation before analysis"
        )
        normalize = st.checkbox("Z-score Normalization", value=True)

# Run analysis button
if st.button("🚀 Run Analysis", type="primary"):
    
    with st.spinner("Running analysis... This may take a few minutes..."):
        
        try:
            # Prepare data
            if "Causality" in analysis_type:
                df_test = df_raw[[id_col, cause_var, effect_var]].dropna()
                st.write(f"**Testing:** {cause_var} → {effect_var}")
            else:
                df_test = df_raw[[id_col, y_col, x_col]].dropna()
                st.write(f"**Testing:** {y_col} vs {x_col}")
            
            st.write(f"**Entities:** {df_test[id_col].nunique()}")
            st.write(f"**Total observations:** {len(df_test)}")
            
            # Run appropriate analysis
            if analysis_type == "Quantile Cointegration Tests":
                st.write("### 📈 Quantile Cointegration Results")
                results = panel_mqcs_test(
                    df_test, id_col, y_col, x_col, quantiles, B, transform, normalize
                )
                
            elif analysis_type == "Quantile Granger Causality Tests":
                st.write("### 🔄 Quantile Granger Causality Results")
                results = panel_quantile_granger_causality(
                    df_test, id_col, cause_var, effect_var, max_lag, quantiles, B, transform, normalize
                )
                
            elif analysis_type == "Traditional Panel Cointegration Tests":
                st.write("### 🌍 Traditional Panel Cointegration Results")
                df_panel = df_test.rename(columns={id_col: 'Entity'})
                results = run_panel_cointegration_tests(df_panel, y_col, x_col)
                
            else:  # Traditional Granger Causality
                st.write("### 🔄 Traditional Granger Causality Results")
                results = traditional_granger_causality(df_test, id_col, cause_var, effect_var, max_lag)
            
            # Display results
            if len(results) > 0:
                st.success(f"✅ Analysis completed successfully!")
                
                # Display appropriate results format
                if analysis_type == "Quantile Cointegration Tests":
                    display_cols = ['Entity', 'N'] + [col for col in results.columns 
                                                    if not col.endswith(('_pval', '_stat')) 
                                                    and col not in ['Entity', 'N']]
                    display_df = results[display_cols].sort_values('Entity')
                    
                elif analysis_type == "Quantile Granger Causality Tests":
                    display_cols = ['Entity', 'Lag', 'Quantile', 'Statistic', 'P-value']
                    display_df = results[display_cols].sort_values(['Entity', 'Lag', 'Quantile'])
                    
                elif analysis_type == "Traditional Granger Causality Tests":
                    display_cols = ['Entity', 'Lag', 'Statistic', 'P-value']
                    display_df = results[display_cols].sort_values(['Entity', 'Lag'])
                    
                else:  # Panel cointegration
                    display_df = results
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download results
                st.write("### 💾 Download Results")
                csv = results.to_csv(index=False)
                st.download_button(
                    "📥 Download Full Results (CSV)",
                    csv,
                    f"{analysis_type.replace(' ', '_').lower()}_results.csv",
                    "text/csv"
                )
                
                # Interpretation
                st.write("### 📖 Interpretation Guide")
                if "Cointegration" in analysis_type:
                    st.markdown("""
                    **Cointegration Test Interpretation:**
                    - **Null Hypothesis**: No cointegration relationship exists
                    - **Significant result** (p < 0.05): Evidence of long-run relationship
                    - **Quantile approach**: Tests relationship at different parts of distribution
                    """)
                else:  # Causality tests
                    st.markdown("""
                    **Granger Causality Interpretation:**
                    - **Null Hypothesis**: X does not Granger-cause Y
                    - **Significant result** (p < 0.05): Evidence of predictive causality
                    - **Quantile approach**: Tests causality at different parts of distribution
                    - **Note**: Granger causality ≠ true causality, but predictive relationship
                    """)
                
            else:
                st.error("❌ No valid results obtained. Check data and settings.")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

# Footer
st.write("---")
st.markdown("""
**Methodology References:**
- **Quantile Cointegration**: Xiao, Z. (2009). *Journal of Econometrics*
- **Quantile Granger Causality**: Troster, V. (2018). *Journal of Financial Econometrics*
- **Traditional Tests**: Pedroni (1999), Kao (1999), Granger (1969)

**Note:** Quantile-based tests provide insights across different parts of the distribution, 
offering more comprehensive analysis than traditional mean-based approaches.
""")
