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

id_col   = st.selectbox("Select panel-ID column", ["index"] + list(df_raw.columns))
date_col = st.selectbox("Select date column (optional)", ["none"] + list(df_raw.columns))
y_col    = st.selectbox("Dependent variable (y)", df_raw.columns)
x_cols   = st.multiselect("Independent variables (x)", df_raw.columns,
                            default=[c for c in df_raw.columns if c not in {y_col, id_col, date_col}])
if not x_cols: st.warning("Choose at least one x variable."); st.stop()

# ----------  tidy panel  -----------------------------------------------------
if id_col == "index":
    wide = df_raw
else:
    idx = [id_col] if date_col=="none" else [id_col, date_col]
    wide = df_raw.set_index(idx)

panel = wide[[y_col] + x_cols].dropna()

# ============ DATA VALIDATION & TRANSFORMATION ============
# Check for non-positive values before log
non_positive = (panel <= 0).any()
if non_positive.any():
    st.error(f"❌ Cannot apply log transformation. Non-positive values found in: {', '.join(non_positive[non_positive].index.tolist())}")
    st.warning("Please ensure all data values are positive (> 0) for log transformation.")
    st.stop()

# Apply log transformation
panel = np.log(panel)

# Check for inf/nan after log
if panel.isin([np.inf, -np.inf]).any().any() or panel.isna().any().any():
    st.error("❌ Invalid values (inf/NaN) detected after log transformation.")
    st.stop()

# Z-score normalization with safety check
panel_normalized = pd.DataFrame(index=panel.index)
for col in panel.columns:
    mean_val = panel[col].mean()
    std_val = panel[col].std()
    
    if std_val == 0 or np.isnan(std_val) or np.isinf(std_val):
        st.error(f"❌ Column '{col}' has zero or invalid standard deviation (constant values). Cannot normalize.")
        st.stop()
    
    panel_normalized[col] = (panel[col] - mean_val) / std_val

panel = panel_normalized.reset_index()

# Final validation
if panel.isin([np.inf, -np.inf]).any().any() or panel.isna().any().any():
    st.error("❌ Invalid values detected in normalized data.")
    st.write("Data summary:", panel.describe())
    st.stop()

chart_ready = panel.select_dtypes(include=np.number)
st.write("### 2. Normalised (log, z-score) series")
st.line_chart(chart_ready)

# ----------  core functions  -------------------------------------------------
def _mqcs(y, x, tau, h=None, B=500, block_size=None, seed=42):
    """MQCS test with data validation"""
    # Validate inputs
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
    
    # Check for multicollinearity
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("Perfect multicollinearity detected in X")
    
    mod = sm.QuantReg(y, X)
    beta = mod.fit(q=tau).params
    u = y - X @ beta
    psi = tau - (u < 0)
    
    # ---- Newey-West long-run variance ----
    gamma = sm.tsa.stattools.acf(psi, nlags=h, fft=False)
    lrv   = gamma[0] + 2 * gamma[1:].sum()
    
    if lrv <= 0:
        lrv = 1e-8  # Prevent division by zero
    
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
            continue  # Skip failed bootstrap iterations
    
    if len(bs_stats) == 0:
        return stat, np.nan
    
    pval = 1 - np.mean(np.array(bs_stats) <= stat)
    return stat, pval

def _eg_adf(y, x):
    """Engle-Granger ADF test"""
    resid = sm.OLS(y, sm.add_constant(x.reshape(-1,1))).fit().resid
    return sm.tsa.adfuller(resid, regression='ct', autolag='AIC')[0]

# ----------  run tests  ------------------------------------------------------
taus = [0.1, 0.3, 0.5, 0.7, 0.9]
results = []
errors = []

for entity, sub in panel.groupby(id_col if id_col != "index" else panel.columns[0]):
    try:
        y = sub[y_col].values
        x1 = sub[x_cols[0]].values
        
        # Validate data
        if len(y) < 10:
            errors.append(f"ID {entity}: insufficient data (n={len(y)})")
            continue
        
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x1)):
            errors.append(f"ID {entity}: contains inf/NaN")
            continue
        
        row = {'ID': entity}
        
        for tau in taus:
            stat, pval = _mqcs(y, x1, tau)
            if np.isnan(pval):
                row[f'MQCS τ={tau}'] = "N/A"
            else:
                stars = ''.join(['*' for t in [0.1,0.05,0.01] if pval < t])
                row[f'MQCS τ={tau}'] = f"{stat:.3f}{stars}"
        
        row['Linear ADF t'] = f"{_eg_adf(y, x1):.3f}"
        results.append(row)
        
    except Exception as e:
        errors.append(f"ID {entity}: {str(e)}")

if errors:
    with st.expander("⚠️ Warnings/Errors during processing"):
        for err in errors:
            st.warning(err)

if not results:
    st.error("❌ No valid results. Please check your data.")
    st.stop()

out = pd.DataFrame(results).set_index('ID')
st.write("### 3. Panel results")
st.dataframe(out)

csv = out.to_csv()
st.download_button("Download CSV", csv, "panel_mqcs.csv", "text/csv")

st.info(
    "**How numbers were calculated**  \n"
    "- Each individual (row) tested separately.  \n"
    "- MQCS(τ): sup-norm of quantile-cointegration residuals; p-value via 500-block-bootstrap (stars 10/5/1%).  \n"
    "- Linear ADF t: Engle-Granger residual ADF t-stat (Ho: no cointegration)."
)
