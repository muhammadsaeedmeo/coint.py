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
panel = np.log(panel).apply(lambda x: (x - x.mean())/x.std())
panel = panel.reset_index()
chart_ready = panel.select_dtypes(include=np.number)      # numeric only for chart
st.write("### 2. Normalised (log, z-score) series")
st.line_chart(chart_ready)

# ----------  core functions  -------------------------------------------------
def _mqcs(y, x, tau, h=None, B=500, block_size=None, seed=42):
    n = len(y)
    h = int(n**(1/5)) if h is None else h
    block_size = int(n**(1/3)) if block_size is None else block_size
    X = sm.add_constant(x.reshape(-1,1))
    mod = sm.QuantReg(y, X)
    beta = mod.fit(q=tau).params
    u = y - X @ beta
    psi = tau - (u < 0)
    lrv = sm.stats.cov_nw(psi, h)                           # official API
    S = np.cumsum(psi)/np.sqrt(n)/np.sqrt(lrv)
    stat = np.max(np.abs(S))
    rng = np.random.default_rng(seed)
    bs_stats = []
    cbb = CircularBlockBootstrap(block_size, x, y, random_state=rng)
    for _, bx, by in cbb.bootstrap(B):
        bX = sm.add_constant(bx.reshape(-1,1))
        bmod = sm.QuantReg(by, bX).fit(q=tau)
        bu = by - bX @ bmod.params
        bpsi = tau - (bu < 0)
        blrv = sm.stats.cov_nw(bpsi, h)                     # official API
        bS = np.cumsum(bpsi)/np.sqrt(len(bpsi))/np.sqrt(blrv)
        bs_stats.append(np.max(np.abs(bS)))
    pval = 1 - np.mean(np.array(bs_stats) <= stat)
    return stat, pval

def _eg_adf(y, x):
    resid = sm.OLS(y, sm.add_constant(x.reshape(-1,1))).fit().resid
    return sm.tsa.adfuller(resid, regression='ct', autolag='AIC')[0]

# ----------  run tests  ------------------------------------------------------
taus = [0.1, 0.3, 0.5, 0.7, 0.9]
results = []
for entity, sub in panel.groupby(id_col if id_col != "index" else panel.columns[0]):
    y = sub[y_col].values
    x1 = sub[x_cols[0]].values
    row = {'ID': entity}
    for tau in taus:
        stat, pval = _mqcs(y, x1, tau)
        stars = ''.join(['*' for t in [0.1,0.05,0.01] if pval < t])
        row[f'MQCS τ={tau}'] = f"{stat:.3f}{stars}"
    row['Linear ADF t'] = f"{_eg_adf(y, x1):.3f}"
    results.append(row)

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
