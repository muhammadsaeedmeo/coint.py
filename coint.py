"""
Streamlit App: Panel MQCS Quantile Cointegration Test
Based on Xiao (2009) methodology
"""

import streamlit as st
import pandas as pd
import numpy as np
from mqcs_cointegration import panel_mqcs_test, MQCSTest, CRITICAL_VALUES

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
