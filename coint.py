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

def panel_mqcs_test(df, id_col, y_col, x_col, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
                    B=1000, transform='log_shift', normalize=True):
    """
    Run MQCS test on panel data - COUNTRY SPECIFIC
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
                row[f'τ={tau:.1f}_stat'] = stat  # Store raw stat for sorting
                
            except Exception as e:
                st.warning(f"Error for {entity} at τ={tau}: {str(e)}")
                row[f'τ={tau:.1f}'] = 'N/A'
                row[f'τ={tau:.1f}_pval'] = np.nan
                row[f'τ={tau:.1f}_stat'] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)

def panel_wide_mqcs_test(df, y_col, x_col, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
                         B=1000, transform='log_shift', normalize=True):
    """
    Run MQCS test on ENTIRE PANEL (pooled data)
    """
    
    # Combine all data
    y_data = df[y_col].values
    x_data = df[x_col].values
    
    # Apply transformation
    try:
        y_data, x_data = _transform_data(y_data, x_data, transform, "Panel")
    except Exception as e:
        raise ValueError(f"Panel transformation error: {str(e)}")
    
    # Apply normalization to entire panel
    if normalize:
        y_data = (y_data - np.mean(y_data)) / np.std(y_data)
        x_data = (x_data - np.mean(x_data)) / np.std(x_data)
    
    # Check validity
    if len(y_data) < 20:
        raise ValueError(f"Insufficient panel data: n={len(y_data)}")
    
    if not np.all(np.isfinite(y_data)) or not np.all(np.isfinite(x_data)):
        raise ValueError("Panel data contains inf/NaN")
    
    # Run tests at each quantile for entire panel
    results = {'Panel': 'All Countries', 'N': len(y_data)}
    
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
            
            results[f'τ={tau:.1f}'] = f"{stat:.3f}{stars}"
            results[f'τ={tau:.1f}_pval'] = pval
            results[f'τ={tau:.1f}_stat'] = stat
            
        except Exception as e:
            st.warning(f"Error for panel at τ={tau}: {str(e)}")
            results[f'τ={tau:.1f}'] = 'N/A'
            results[f'τ={tau:.1f}_pval'] = np.nan
            results[f'τ={tau:.1f}_stat'] = np.nan
    
    return pd.DataFrame([results])

# =============================================================================
# PANEL COINTEGRATION TESTS
# =============================================================================

def pedroni_panel_cointegration(df, y_col, x_col):
    """
    Pedroni (1999) panel cointegration test
    Tests for cointegration in panel data using residual-based approach
    """
    try:
        # Group by entity and run individual regressions
        entities = df['Entity'].unique() if 'Entity' in df.columns else df['Country'].unique()
        residuals_list = []
        n_entities = len(entities)
        
        for entity in entities:
            entity_data = df[df['Entity' if 'Entity' in df.columns else 'Country'] == entity]
            if len(entity_data) < 10:
                continue
                
            y = entity_data[y_col].values
            x = entity_data[x_col].values
            
            # Run OLS regression
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            results = model.fit()
            residuals = results.resid
            
            residuals_list.append(residuals)
        
        if len(residuals_list) < 2:
            return {'Statistic': np.nan, 'P-value': np.nan, 'Test': 'Pedroni (Insufficient data)'}
        
        # Calculate panel cointegration statistic (simplified version)
        # This is a simplified implementation - actual Pedroni test is more complex
        all_residuals = np.concatenate(residuals_list)
        adf_stat, p_value, _, _, _ = sm.tsa.adfuller(all_residuals)
        
        return {
            'Statistic': adf_stat,
            'P-value': p_value,
            'Test': 'Pedroni Panel Cointegration',
            'Interpretation': 'Reject null of no cointegration if p-value < 0.05'
        }
        
    except Exception as e:
        return {'Statistic': np.nan, 'P-value': np.nan, 'Test': f'Pedroni (Error: {str(e)})'}

def kao_panel_cointegration(df, y_col, x_col):
    """
    Kao (1999) panel cointegration test
    Another residual-based panel cointegration test
    """
    try:
        entities = df['Entity'].unique() if 'Entity' in df.columns else df['Country'].unique()
        t_stats = []
        
        for entity in entities:
            entity_data = df[df['Entity' if 'Entity' in df.columns else 'Country'] == entity]
            if len(entity_data) < 10:
                continue
                
            y = entity_data[y_col].values
            x = entity_data[x_col].values
            
            # Run OLS regression
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            results = model.fit()
            residuals = results.resid
            
            # ADF test on residuals
            adf_stat, p_value, _, _, _ = sm.tsa.adfuller(residuals)
            t_stats.append(adf_stat)
        
        if len(t_stats) < 2:
            return {'Statistic': np.nan, 'P-value': np.nan, 'Test': 'Kao (Insufficient data)'}
        
        # Kao test statistic (simplified)
        mean_t_stat = np.mean(t_stats)
        # Simplified p-value calculation
        p_value = 1 - stats.norm.cdf(abs(mean_t_stat))
        
        return {
            'Statistic': mean_t_stat,
            'P-value': p_value,
            'Test': 'Kao Panel Cointegration',
            'Interpretation': 'Reject null of no cointegration if p-value < 0.05'
        }
        
    except Exception as e:
        return {'Statistic': np.nan, 'P-value': np.nan, 'Test': f'Kao (Error: {str(e)})'}

def fisher_combined_cointegration(df, y_col, x_col):
    """
    Fisher's combined test for panel cointegration
    Combines p-values from individual country cointegration tests
    """
    try:
        entities = df['Entity'].unique() if 'Entity' in df.columns else df['Country'].unique()
        p_values = []
        
        for entity in entities:
            entity_data = df[df['Entity' if 'Entity' in df.columns else 'Country'] == entity]
            if len(entity_data) < 10:
                continue
                
            y = entity_data[y_col].values
            x = entity_data[x_col].values
            
            # ADF test on residuals from OLS regression
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            results = model.fit()
            residuals = results.resid
            
            _, p_value, _, _, _ = sm.tsa.adfuller(residuals)
            p_values.append(p_value)
        
        if len(p_values) < 2:
            return {'Statistic': np.nan, 'P-value': np.nan, 'Test': 'Fisher (Insufficient data)'}
        
        # Fisher's combined test
        chi2_stat = -2 * np.sum(np.log(p_values))
        df = 2 * len(p_values)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        return {
            'Statistic': chi2_stat,
            'P-value': p_value,
            'Test': 'Fisher Combined Test',
            'Interpretation': 'Reject null of no cointegration if p-value < 0.05'
        }
        
    except Exception as e:
        return {'Statistic': np.nan, 'P-value': np.nan, 'Test': f'Fisher (Error: {str(e)})'}

def run_panel_cointegration_tests(df, y_col, x_col):
    """
    Run multiple panel cointegration tests
    """
    st.write("### 🌍 Panel Cointegration Tests")
    st.write("Testing for cointegration in the entire panel using different methodologies:")
    
    results = []
    
    # Run different panel cointegration tests
    tests = [
        pedroni_panel_cointegration,
        kao_panel_cointegration,
        fisher_combined_cointegration
    ]
    
    for test_func in tests:
        result = test_func(df, y_col, x_col)
        results.append(result)
    
    # Create results dataframe
    panel_results = pd.DataFrame(results)
    
    # Add significance stars
    def add_stars(pval):
        if pd.isna(pval):
            return 'N/A'
        if pval < 0.01:
            return f"{pval:.4f}***"
        elif pval < 0.05:
            return f"{pval:.4f}**"
        elif pval < 0.10:
            return f"{pval:.4f}*"
        else:
            return f"{pval:.4f}"
    
    panel_results['P-value'] = panel_results['P-value'].apply(add_stars)
    
    # Display results
    st.dataframe(panel_results[['Test', 'Statistic', 'P-value', 'Interpretation']], 
                 use_container_width=True)
    
    return panel_results

# Critical values from Xiao and Phillips (2002) for constant coefficient case
CRITICAL_VALUES = {
    0.10: 1.616,  # 10% significance
    0.05: 1.842,  # 5% significance
    0.01: 2.326   # 1% significance
}

# =============================================================================
# STREAMLIT APP
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

# Load data with better error handling
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            # Try different Excel engines
            try:
                return pd.read_excel(file, engine='openpyxl')
            except:
                try:
                    return pd.read_excel(file, engine='xlrd')
                except:
                    st.error("Cannot read Excel file. Please ensure openpyxl or xlrd is installed.")
                    st.stop()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

try:
    df_raw = load_data(uploaded)
except ImportError as e:
    st.error("""
    **Missing Dependencies Error**
    
    The app cannot read Excel files because required packages are missing.
    
    **Please add these to your requirements.txt:**
    ```
    openpyxl
    ```
    
    If you're using Streamlit Cloud, make sure your requirements.txt includes openpyxl.
    """)
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

st.write("### 📁 Raw Data Preview")
st.dataframe(df_raw.head(10))
st.write(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# Analysis type selection
st.write("### 🔄 Analysis Type")
analysis_type = st.radio(
    "Select analysis type:",
    ["Country-specific Quantile Cointegration", "Panel-wide Quantile Cointegration", "Panel Cointegration Tests"],
    help="""
    - Country-specific: Test each country separately using quantile cointegration
    - Panel-wide: Test all countries together as one pooled sample using quantile cointegration  
    - Panel Cointegration Tests: Traditional panel cointegration tests (Pedroni, Kao, Fisher)
    """
)

# Column selection
st.write("### ⚙️ Configure Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    # Auto-detect entity column
    possible_entity_cols = [col for col in df_raw.columns if df_raw[col].dtype == 'object' and df_raw[col].nunique() < 100]
    id_col = st.selectbox("Entity ID Column (e.g., Country)", possible_entity_cols if possible_entity_cols else df_raw.columns)

with col2:
    y_col = st.selectbox("Dependent Variable (y)", 
                         [c for c in df_raw.columns if c != id_col])

with col3:
    x_col = st.selectbox("Independent Variable (x)", 
                         [c for c in df_raw.columns if c not in [id_col, y_col]])

# Only show transformation settings for quantile cointegration tests
if analysis_type != "Panel Cointegration Tests":
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
if st.button("🚀 Run Analysis", type="primary"):
    
    with st.spinner("Running analysis... This may take a few minutes..."):
        
        try:
            # Prepare data
            df_test = df_raw[[id_col, y_col, x_col]].dropna()
            
            st.write(f"**Entities:** {df_test[id_col].nunique()}")
            st.write(f"**Total observations:** {len(df_test)}")
            
            # Run appropriate test based on analysis type
            if analysis_type == "Country-specific Quantile Cointegration":
                st.write("### 🏴‍☠️ Country-Specific Quantile Cointegration Results")
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
                
                st.success(f"✅ Successfully tested {len(results)} entities")
                
            elif analysis_type == "Panel-wide Quantile Cointegration":
                st.write("### 🌍 Panel-Wide Quantile Cointegration Results")
                results = panel_wide_mqcs_test(
                    df_test,
                    y_col=y_col,
                    x_col=x_col,
                    quantiles=quantiles,
                    B=B,
                    transform=transform,
                    normalize=normalize
                )
                
                st.success("✅ Successfully tested entire panel")
                
            else:  # Panel Cointegration Tests
                # Standardize column names for panel tests
                df_panel = df_test.copy()
                if id_col != 'Entity' and id_col != 'Country':
                    df_panel = df_panel.rename(columns={id_col: 'Entity'})
                
                panel_results = run_panel_cointegration_tests(df_panel, y_col, x_col)
                results = None
            
            # Display results for quantile cointegration tests
            if analysis_type != "Panel Cointegration Tests":
                st.write("## 📊 Results")
                
                # Handle different column names for different analysis types
                if analysis_type == "Country-specific Quantile Cointegration":
                    # For country-specific, we have 'Entity' column
                    display_cols = ['Entity', 'N'] + [col for col in results.columns if not col.endswith('_pval') and not col.endswith('_stat') and col not in ['Entity', 'N']]
                    display_df = results[display_cols].copy()
                    display_df = display_df.sort_values('Entity')
                else:
                    # For panel-wide, we have 'Panel' column instead of 'Entity'
                    display_cols = ['Panel', 'N'] + [col for col in results.columns if not col.endswith('_pval') and not col.endswith('_stat') and col not in ['Panel', 'N']]
                    display_df = results[display_cols].copy()
                    # Rename 'Panel' to 'Entity' for consistent display
                    display_df = display_df.rename(columns={'Panel': 'Entity'})
                
                st.dataframe(display_df, use_container_width=True)
                
                # Interpretation guide for quantile cointegration
                st.write("### 📖 Interpretation Guide")
                st.markdown(f"""
                **Test Statistic Interpretation:**
                - **Larger values** indicate **rejection** of the null hypothesis (no cointegration)
                - **Significance levels:** *** p<0.01, ** p<0.05, * p<0.10
                - Test statistic > critical value ⇒ Reject null of no cointegration
                
                **Critical Values (Xiao & Phillips 2002):**
                - 10% level: {CRITICAL_VALUES[0.10]}
                - 5% level: {CRITICAL_VALUES[0.05]}
                - 1% level: {CRITICAL_VALUES[0.01]}
                
                **Methodology:**
                - **Quantile Cointegration Test** based on Xiao (2009)
                - **MQCS Statistic**: Modified Quantile Cointegration Statistic
                - **Bootstrap**: {B} moving block bootstrap replications for p-values
                - **Transformation**: {transform}
                - **Normalization**: {'Yes' if normalize else 'No'}
                """)
                
                # Full results with p-values
                with st.expander("📋 Full Results (including p-values)"):
                    if analysis_type == "Country-specific Quantile Cointegration":
                        full_display_cols = ['Entity', 'N'] + [col for col in results.columns if not col.endswith('_stat') and col not in ['Entity', 'N']]
                        full_display_df = results[full_display_cols].copy()
                        full_display_df = full_display_df.sort_values('Entity')
                    else:
                        full_display_cols = ['Panel', 'N'] + [col for col in results.columns if not col.endswith('_stat') and col not in ['Panel', 'N']]
                        full_display_df = results[full_display_cols].copy()
                        full_display_df = full_display_df.rename(columns={'Panel': 'Entity'})
                    
                    st.dataframe(full_display_df, use_container_width=True)
                
                # Download buttons for quantile cointegration
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
                    # For download, use the original results without renaming
                    csv_full = results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Results with P-values (CSV)",
                        csv_full,
                        "mqcs_results_full.csv",
                        "text/csv"
                    )
                
                # Summary statistics for country-specific analysis
                if analysis_type == "Country-specific Quantile Cointegration":
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
                    
                    if sig_counts:
                        sig_df = pd.DataFrame(sig_counts).T
                        sig_df.columns = ['Sig at 1%', 'Sig at 5%', 'Sig at 10%']
                        
                        st.write("**Number of entities with significant cointegration:**")
                        st.dataframe(sig_df)
            
            # Results interpretation
            st.write("### 🔍 Results Interpretation")
            if analysis_type == "Country-specific Quantile Cointegration":
                st.markdown(f"""
                **Country-Specific Analysis:**
                - Each country is tested **independently** for quantile cointegration
                - Results show whether **individual countries** exhibit cointegration at different quantiles
                - Useful for identifying **country-specific** cointegration patterns
                - **Total countries analyzed**: {len(results)}
                """)
            elif analysis_type == "Panel-wide Quantile Cointegration":
                st.markdown(f"""
                **Panel-Wide Analysis:**
                - All countries are **pooled together** and tested as one sample
                - Results show whether the **entire panel** exhibits cointegration at different quantiles
                - Useful for identifying **overall** cointegration relationships
                - **Total observations**: {results.iloc[0]['N']}
                - **Interpretation**: The test examines if there's a long-run relationship between {y_col} and {x_col} across the entire panel
                """)
            else:
                st.markdown(f"""
                **Panel Cointegration Tests:**
                - **Pedroni Test**: Residual-based panel cointegration test
                - **Kao Test**: Another residual-based panel cointegration test  
                - **Fisher Test**: Combined test from individual country cointegration tests
                - **Interpretation**: Reject null hypothesis of no cointegration if p-value < 0.05
                - These tests provide **overall panel-level** evidence of cointegration
                """)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

# Footer
st.write("---")
st.markdown("""
**Reference:**
- Xiao, Z. (2009). "Quantile Cointegrating Regression." *Journal of Econometrics*.
- Pedroni, P. (1999). "Critical values for cointegration tests in heterogeneous panels with multiple regressors."
- Kao, C. (1999). "Spurious regression and residual-based tests for cointegration in panel data."
- Test implements moving block bootstrap for p-value computation
- Entity-specific transformation ensures valid panel cointegration tests

**Note:** The MQCS test examines cointegration across different parts of the conditional distribution (quantiles), 
providing a more comprehensive view of the relationship between variables than traditional mean-based cointegration tests.
""")
