import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from mqcs import mqcs, _zscore

st.set_page_config(page_title="MQCS quantile-cointegration tester", layout="wide")
st.title("📊 MQCS(τ) quantile-cointegration test")

countries = st.sidebar.text_area("Yahoo ticker list (one per line)",
                                 "CNY=X\nJPY=X\nKRW=X").split()
quantiles = st.sidebar.text_input("Quantile grid", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
taus = list(map(float, quantiles.split(",")))
B = st.sidebar.slider("Bootstrap replicas", 200, 5000, 1000, 200)
block_size = st.sidebar.number_input("Block size (0 = auto)", 0, 100, 0)

@st.cache_data
def load(tickers):
    df = yf.download(tickers, start="2010-01-01")['Adj Close'].dropna()
    return df

prices = load(countries)
st.write("### 1. Raw prices (last 5 rows)")
st.dataframe(prices.tail())

lprices = np.log(prices).apply(_zscore)
st.write("### 2. Log & z-scored series")
st.line_chart(lprices)

st.write("### 3. Running MQCS(τ) …")
y_name = st.selectbox("Choose dependent variable (y)", lprices.columns)
x_names = [c for c in lprices.columns if c != y_name]

table = []
for x_name in x_names:
    y = lprices[y_name].values
    x = lprices[x_name].values
    row = {'pair': f"{y_name} ~ {x_name}"}
    for tau in taus:
        res = mqcs(y, x, tau, B=B, block_size=int(block_size) if block_size else None)
        stars = ''.join(['*' for t in [0.1, 0.05, 0.01] if res['pval'] < t])
        row[f"τ={tau}"] = f"{res['stat']:.3f}{stars}"
    table.append(row)

df_out = pd.DataFrame(table).set_index('pair')
st.write("### 4. MQCS(τ) table (stars 10/5/1%)")
st.dataframe(df_out)

csv = df_out.to_csv()
st.download_button("Download CSV", csv, "mqcs_results.csv", "text/csv")
