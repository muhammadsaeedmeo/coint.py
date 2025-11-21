"""
Streamlit app: panel MQCS(τ) quantile-cointegration test + linear cointegration check
Author: you
License: MIT
"""
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch.bootstrap import CircularBlockBootstrap

st.set_page_config(page_title="Panel MQCS test", layout="wide")
st.title("📊 Panel-data MQCS(τ) quantile-cointegration test")

uploaded = st.file_uploader("Upload panel CSV or Excel", type=["csv","xlsx"])
if uploaded is None: st.stop()

@st.cache_data
def read(f):
    return pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)

df_raw = read(uploaded)
st.write("### 1. Raw data (first 5 rows)")
st.dataframe(df_raw.head())

id_col   = st.selectbox("Select panel-ID column (e.g., Country)", ["index"] + list(df_raw.columns))
date_col = st.selectbox("Select date column (optional)", ["none"] + list(df_raw.columns))
y_col    = st.selectbox("Dependent variable (y)", df_raw.columns)
x_cols   = st.multiselect("Independent variables (x)", df_raw.columns,
                            default=[c for c in df_raw.columns if c not in {y_col, id_col, date_col}])
if not x_cols: st.warning("Choose at least one x variable."); st.stop()

# ========== TRANSFORMATION OPTIONS ==========
st.write("### 🔧 Transformation Settings")
col1, col2 = st.columns(2)
with col1:
    transform_method = st.selectbox(
        "Transformation method",
        ["Log with min-shift (entity-specific)", 
         "IHS - Inverse Hyperbolic Sine",
         "Log only (fails if negative)",
         "No transformation (use raw data)"],
        help="Entity-specific means each country/entity is transformed separately"
    )
with col2:
    normalize = st.checkbox("Apply z-score normalization (within entity)", value=True,
                           help="Standardize each entity's series to mean=0, std=1")

# ----------  tidy panel  -----------------------------------------------------
if id_col == "index":
    wide = df_raw.copy()
    wide['__entity__'] = wide.index
    entity_col = '__entity__'
else:
    entity_col = id_col
    idx = [id_col] if date_col=="none" else [id_col, date_col]
    wide = df_raw.set_index(idx).reset_index()

panel = wide[[entity_col] + [y_col] + x_cols].dropna()

st.write(f"**Data check**: {panel[entity_col].nunique()} entities, {len(panel)} total observations")

# ========== ENTITY-SPECIFIC TRANSFORMATION ==========
def transform_series(series, method, entity_name, var_name):
    """Apply transformation to a single series with diagnostics"""
    original_series = series.copy()
    
    if method == "Log with min-shift (entity-specific)":
        min_val = series.min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            series = series + shift
            st.info(f"🔄 {entity_name} | {var_name}: shifted by +{shift:.2f} (min was {min_val:.2f})")
        series = np.log(series)
        
    elif method == "IHS - Inverse Hyperbolic Sine":
        # arcsinh(x) = log(x + sqrt(x^2 + 1))
        series = np.arcsinh(series)
        
    elif method == "Log only (fails if negative)":
        if (series <= 0).any():
            raise ValueError(f"{entity_name} | {var_name}: has non-positive values. Use min-shift or IHS.")
        series = np.log(series)
        
    # else: No transformation
    
    return series

# Apply transformation entity by entity
transformed_data = []
transformation_log = []

for entity, entity_df in panel.groupby(entity_col):
    entity_transformed = entity_df[[entity_col]].copy()
    
    # Transform each variable
    for var in [y_col] + x_cols:
        try:
            entity_transformed[var] = transform_series(
                entity_df[var].values, 
                transform_method,
                entity,
                var
            )
        except Exception as e:
            st.error(f"❌ {entity} | {var}: {str(e)}")
            st.stop()
    
    # Check for inf/nan after transformation
    if entity_transformed[[y_col] + x_cols].isin([np.inf, -np.inf]).any().any():
        st.error(f"❌ {entity}: inf values after transformation")
        st.stop()
    
    if entity_transformed[[y_col] + x_cols].isna().any().any():
        st.error(f"❌ {entity}: NaN values after transformation")
        st.stop()
    
    # Z-score normalization (within entity)
    if normalize:
        for var in [y_col] + x_cols:
            mean_val = entity_transformed[var].mean()
            std_val = entity_transformed[var].std()
            
            if std_val == 0 or not np.isfinite(std_val):
                st.error(f"❌ {entity} | {var}: zero or invalid std (constant values)")
                st.stop()
            
            entity_transformed[var] = (entity_transformed[var] - mean_val) / std_val
    
    transformed_data.append(entity_transformed)

panel_transformed = pd.concat(transformed_data, ignore_index=True)

# Prepare for visualization
if date_col != "none":
    # Add date back for better chart
    panel_transformed = panel_transformed.merge(
        panel[[entity_col, date_col]], 
        left_index=True, 
        right_index=True, 
        how='left'
    )

st.write("### 2. Transformed series")
st.write(f"**Method**: {transform_method}")
st.write(f"**Normalized**: {'Yes (z-score within entity)' if normalize else 'No'}")

chart_ready = panel_transformed.select_dtypes(include=np.number)
st.line_chart(chart_ready[[y_col] + x_cols])

# Show summary statistics
with st.expander("📊 Summary statistics (post-transformation)"):
    st.dataframe(panel_transformed.groupby(entity_col)[[y_col] + x_cols].describe().T)

# ----------  core functions  -------------------------------------------------
def _mqcs(y, x, tau, h=None, B=500, block_size=None, seed=42):
    """MQCS test with data validation"""
    if len(y) != len(x):
        raise ValueError(f"Length mismatch: y={len(y)}, x={len(x)}")
    
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(x)):
        raise ValueError("Input contains inf or NaN values")
    
    n = len(y)
    if n < 10:
        raise ValueError(f"Insufficient data: n={n}")
    
    h = int(n**(1/5)) if h is None else h
    block_size = int(n**(1/3)) if block_size is None else block_size
    
    X = sm.add_constant(x.reshape(-1,1))
    
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("Perfect multicollinearity detected in X")
    
    mod = sm.QuantReg(y, X)
    beta = mod.fit(q=tau).params
    u = y - X @ beta
    psi = tau - (u < 0)
    
    # Newey-West long-run variance
    gamma = sm.tsa.stattools.acf(psi, nlags=h, fft=False)
    lrv   = gamma[0] + 2 * gamma[1:].sum()
    
    if lrv <= 0:
        lrv = 1e-8
    
    S = np.cumsum(psi)/np.sqrt(n)/np.sqrt(lrv)
    stat = np.max(np.abs(S))
    
    # Bootstrap
    rng = np.random.default_rng(seed)
    bs_stats = []
    cbb = CircularBlockBootstrap(block_size, x, y, random_state=rng)
    
    for _, bx, by in cbb.bootstrap(B):
        try:
            bX = sm.add_constant(bx.reshape(-1,1))
            bmod = sm.QuantReg(by, bX).fit(q=tau)
            bu = by - bX @ bmod.params
            bpsi = tau - (bu < 0)
            bgamma = sm.tsa.stattools.acf(bpsi, nlags=h, fft=False)
            blrv   = bgamma[0] + 2 * bgamma[1:].sum()
            
            if blrv <= 0:
                blrv = 1e-8
            
            bS = np.cumsum(bpsi)/np.sqrt(len(bpsi))/np.sqrt(blrv)
            bs_stats.append(np.max(np.abs(bS)))
        except:
            continue
    
    if len(bs_stats) == 0:
        return stat, np.nan
    
    pval = 1 - np.mean(np.array(bs_stats) <= stat)
    return stat, pval

def _eg_adf(y, x):
    """Engle-Granger ADF test"""
    resid = sm.OLS(y, sm.add_constant(x.reshape(-1,1))).fit().resid
    return sm.tsa.adfuller(resid, regression='ct', autolag='AIC')[0]

# ----------  run tests  ------------------------------------------------------
st.write("### 3. Running tests...")
progress_bar = st.progress(0)
taus = [0.1, 0.3, 0.5, 0.7, 0.9]
results = []
errors = []

entities = panel_transformed[entity_col].unique()
for idx, entity in enumerate(entities):
    try:
        sub = panel_transformed[panel_transformed[entity_col] == entity]
        y = sub[y_col].values
        x1 = sub[x_cols[0]].values
        
        if len(y) < 10:
            errors.append(f"ID {entity}: insufficient data (n={len(y)})")
            continue
        
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x1)):
            errors.append(f"ID {entity}: contains inf/NaN")
            continue
        
        row = {'ID': entity, 'N': len(y)}
        
        for tau in taus:
            stat, pval = _mqcs(y, x1, tau)
            if np.isnan(pval):
                row[f'τ={tau}'] = "N/A"
            else:
                stars = ''.join(['*' for t in [0.1,0.05,0.01] if pval < t])
                row[f'τ={tau}'] = f"{stat:.3f}{stars} (p={pval:.3f})"
        
        row['ADF-t'] = f"{_eg_adf(y, x1):.3f}"
        results.append(row)
        
    except Exception as e:
        errors.append(f"ID {entity}: {str(e)}")
    
    progress_bar.progress((idx + 1) / len(entities))

progress_bar.empty()

if errors:
    st.error(f"⚠️ {len(errors)} errors occurred during processing:")
    for err in errors:
        st.error(err)

if not results:
    st.error("❌ No valid results. Please check your data and errors above.")
    st.write("**Debug Info:**")
    st.write(f"- Total entities in data: {len(entities)}")
    st.write(f"- Entities processed: {len(results)}")
    st.write(f"- Errors encountered: {len(errors)}")
    st.write("\n**Transformed data sample:**")
    st.dataframe(panel_transformed.head(20))
    st.stop()

out = pd.DataFrame(results).set_index('ID')
st.write("### 4. Panel MQCS Results")
st.dataframe(out, use_container_width=True)

csv = out.to_csv()
st.download_button("📥 Download Results CSV", csv, "panel_mqcs_results.csv", "text/csv")

st.info(
    "**Interpretation Guide**  \n"
    "- **MQCS(τ)**: Quantile cointegration test statistic at quantile τ  \n"
    "- **Stars**: *** p<0.01, ** p<0.05, * p<0.10 (reject H₀: no cointegration)  \n"
    "- **ADF-t**: Engle-Granger residual ADF t-statistic (more negative = stronger cointegration)  \n"
    "- Each entity tested separately with entity-specific transformation  \n"
    "- Bootstrap iterations: 500 with circular block bootstrap"
)

st.success("✅ Analysis complete!")
