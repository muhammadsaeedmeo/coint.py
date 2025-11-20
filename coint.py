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
import io

# --------------------  FRONT-END  -------------------------------------------
st.set_page_config(page_title="Panel MQCS test", layout="wide")
st.title("📊 Panel-data MQCS(τ) quantile-cointegration test")

# ----------  DATA UPLOAD  ----------------------------------------------------
uploaded = st.file_uploader("Upload panel CSV or Excel", type=["csv","xlsx"])
if uploaded is None: st.stop()

@st.cache_data
def read(f):
    return pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)

df_raw = read(uploaded)
st.write("### 1. Raw data (first 5 rows)")
st.dataframe(df_raw.head())

# ----------  COLUMN SELECTION  ----------------------------------------------
id_col   = st.selectbox("Select panel-ID column", ["index"] + list(df_raw.columns))
date_col = st.selectbox("Select date column (optional)", ["none"] + list(df_raw.columns))
y_col    = st.selectbox("Dependent variable (y)", df_raw.columns)
x_cols   = st.multiselect("Independent variables (x)", df_raw.columns,
                            default=[c for c in df_raw.columns if c not in {y_col, id_col, date_col}])
if not x_cols:
    st.warning("Choose at least one x variable."); st.stop()

# ----------  TIDY PANEL PREP  ------------------------------------------------
if id_col == "index":
    wide = df_raw
else:
    idx = [id_col] if date_col=="none" else [id_col, date_col]
    wide = df_raw.set_index(idx)

panel = wide[[y_col] + x_cols].dropna()
panel = np.log(panel).apply(lambda x: (x - x.mean())/x.std())  # z-score
panel = panel.reset_index()                                    # flatten MultiIndex → simple columns
st.write("### 2. Normalised (log, z-score) series")
st.line_chart(panel.set_index(panel.columns[0]) if panel.columns[0] != id_col else panel.set_index(panel.columns[:2]))

# ----------  CORE FUNCTIONS  -----------------------------------------------
def _mqcs(y, x, tau, h=None, B=500, block_size=None, seed=42):
    n = len(y)
    h = int(n**(1/5)) if h is None else h
    block_size = int(n**(1/3)) if block_size is None else block_size
    X = sm.add_constant(x.reshape(-1,1))
    mod = sm.QuantReg(y, X)
    beta = mod.fit(q=tau).params
    u = y - X @ beta
    psi = tau - (u < 0)
    lrv = sm.stats.sandwich_covariance.cov_nw(psi, h)
    S = np.cumsum(psi)/np.sqrt(n)/np.sqrt(lrv)
    stat = np.max(np.abs(S))
    # bootstrap p-value
    rng = np.random.default_rng(seed)
    bs_stats = []
    cbb = CircularBlockBootstrap(block_size, x, y, random_state=rng)
    for _, bx, by in cbb.bootstrap(B):
        bX = sm.add_constant(bx.reshape(-1,1))
        bmod = sm.QuantReg(by, bX).fit(q=tau)
        bu = by - bX @ bmod.params
        bpsi = tau - (bu < 0)
        blrv = sm.stats.sandwich_covariance.cov_nw(bpsi, h)
        bS = np.cumsum(bpsi)/np.sqrt(len(bpsi))/np.sqrt(blrv)
        bs_stats.append(np.max(np.abs(bS)))
    pval = 1 - np.mean(np.array(bs_stats) <= stat)
    return stat, pval

def _eg_ols(y, x):
    """Engle-Granger residual-based ADF t-stat (single-equation linear cointegration)"""
    X = sm.add_constant(x.reshape(-1,1))
    resid = sm.OLS(y, X).fit().resid
    adf = sm.tsa.adfuller(resid, regression='ct', autolag='AIC')
    return adf[0]   # ADF t-stat

# ----------  RUN TESTS  ------------------------------------------------------
taus = [0.1, 0.3, 0.5, 0.7, 0.9]
results = []

for entity, sub in panel.groupby(id_col if id_col != "index" else panel.columns[0]):
    y = sub[y_col].values
    x = sub[x_cols].values          # allows many x; here we use only the 1st for MQCS
    x1 = x[:,0]                     # first x for MQCS
    # MQCS row
    mqcs_row = {'ID': entity}
    for tau in taus:
        stat, pval = _mqcs(y, x1, tau)
        stars = ''.join(['*' for t in [0.1,0.05,0.01] if pval < t])
        mqcs_row[f'MQCS τ={tau}'] = f"{stat:.3f}{stars}"
    # Linear cointegration (Engle-Granger ADF on residuals)
    adf_t = _eg_ols(y, x1)
    mqcs_row['Linear ADF t'] = f"{adf_t:.3f}"
    results.append(mqcs_row)

out = pd.DataFrame(results).set_index('ID')
st.write("### 3. Panel results")
st.dataframe(out)

# ----------  DOWNLOAD  -------------------------------------------------------
csv = out.to_csv()
st.download_button("Download CSV", csv, "panel_mqcs.csv", "text/csv")

# ----------  HOW NUMBERS WERE CALCULATED  ------------------------------------
st.info(
    "**How the numbers were calculated**  \n"
    "- Each individual (row) is tested separately.  \n"
    "- MQCS(τ): Kolmogorov-Smirnov-type sup-norm statistic of quantile-cointegration residuals; p-values by 500-block-bootstrap (stars 10/5/1%).  \n"
    "- Linear ADF t: Engle-Granger residual-based ADF t-statistic (Ho: no cointegration); critical values ≈ -3.45 (5%) for n≈100."
)
