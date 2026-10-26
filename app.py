import streamlit as st
import pandas as pd
import numpy as np
import io
import json

from services.data_loader import load_data, analyze_schema
from services.synthesizer import TabularSynthesizer
from services.evaluator import evaluate_datasets, evaluate_ml_utility
from services.sample_datasets import generate_sample_dataset
from utils.charts import plot_distributions_comparison, plot_correlation_heatmaps, plot_interactive_eda
from components.ui import apply_custom_theme, render_metric_card

st.set_page_config(
    page_title="Synthetic Tabular Data Generator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()

# Title and subtitle
st.markdown('<div class="badge">PRODUCTION ENGINE • STATISTICALLY FAITHFUL</div>', unsafe_allow_html=True)
st.title("⚡ Production Synthetic Tabular Data Generator")
st.markdown("""
**What is this app?**  
This tool creates realistic, privacy-safe synthetic copies of your tabular datasets without exposing any sensitive or private information.

**How to use it:**
1. **Load Data**: Choose a built-in benchmark dataset or upload your own CSV/Excel file in the left sidebar.
2. **Configure & Generate**: Pick a generation model (Copula, CTGAN, GMM, or KDE) and the number of rows you want, then click **Train & Generate**.
3. **Inspect & Export**: Check data quality, privacy checks, and ML accuracy scores across the tabs, then download your ready-to-use synthetic CSV.
""")
st.markdown("---")

# Initialize session state keys
if "real_df" not in st.session_state:
    st.session_state.real_df = None
if "synth_df" not in st.session_state:
    st.session_state.synth_df = None
if "schema" not in st.session_state:
    st.session_state.schema = {}
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "ml_results" not in st.session_state:
    st.session_state.ml_results = None

# Sidebar controls
with st.sidebar:
    st.header("📂 1. Data Ingestion")
    data_source = st.radio(
        "Choose Data Source:",
        ["Built-in Benchmark", "Upload File (CSV/Excel)"],
        index=0
    )

    if data_source == "Built-in Benchmark":
        sample_choice = st.selectbox(
            "Select Benchmark Dataset:",
            ["customer_churn", "medical_vitals", "financial_fraud"],
            format_func=lambda x: {
                "customer_churn": "📊 Customer Churn (Mixed Dtypes)",
                "medical_vitals": "🩺 Medical Vitals (Clinical)",
                "financial_fraud": "💳 Financial Fraud (Skewed)"
            }.get(x, x)
        )
        n_sample_rows = st.slider("Benchmark Rows:", min_value=50, max_value=1000, value=250, step=50)
        if st.button("Load Benchmark Dataset", use_container_width=True, type="primary"):
            st.session_state.real_df = generate_sample_dataset(sample_choice, n_rows=n_sample_rows)
            st.session_state.schema = analyze_schema(st.session_state.real_df)
            st.session_state.synth_df = None
            st.session_state.eval_results = None
            st.session_state.ml_results = None
            st.success(f"Loaded {len(st.session_state.real_df)} rows!")
    else:
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                st.session_state.real_df = load_data(uploaded_file)
                st.session_state.schema = analyze_schema(st.session_state.real_df)
                st.session_state.synth_df = None
                st.session_state.eval_results = None
                st.session_state.ml_results = None
                st.success(f"Successfully loaded {len(st.session_state.real_df)} rows and {len(st.session_state.real_df.columns)} columns.")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    st.markdown("---")
    st.header("⚙️ 2. Synthesis Configuration")
    method = st.selectbox(
        "Synthesis Algorithm:",
        ["copula", "ctgan", "gmm", "kde"],
        format_func=lambda x: {
            "copula": "Gaussian Copula (Correlation Preserving)",
            "ctgan": "CTGAN (Conditional GAN - Deep Learning)",
            "gmm": "Gaussian Mixture Model (Multi-modal)",
            "kde": "Kernel Density Estimation (Non-parametric)"
        }.get(x, x)
    )
    
    ctgan_epochs = 10
    if method == "ctgan":
        ctgan_epochs = st.slider("CTGAN Training Epochs:", min_value=2, max_value=100, value=15, step=5)

    n_generate = st.number_input("Number of Synthetic Rows:", min_value=10, max_value=50000, value=300, step=50)
    seed = st.number_input("Random Seed:", min_value=0, max_value=999999, value=42)

# Main Application Layout
tabs = st.tabs([
    "📊 Real Data Profiling & EDA",
    "🧬 Synthetic Generator",
    "📈 Quality & Statistical Fidelity",
    "🔒 Privacy & Nearest Neighbor",
    "🤖 ML Utility (TSTR)",
    "💾 Export Data & Reports"
])

# TAB 1: REAL DATA PROFILING & EDA
with tabs[0]:
    if st.session_state.real_df is None:
        st.info("👈 Please select a benchmark dataset or upload your data from the sidebar to get started.")
    else:
        df = st.session_state.real_df
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Total Rows", f"{len(df):,}")
        with c2:
            render_metric_card("Total Columns", f"{len(df.columns):,}")
        with c3:
            num_count = sum(1 for v in st.session_state.schema.values() if v["type"] == "numerical")
            render_metric_card("Numerical Features", f"{num_count}")
        with c4:
            cat_count = sum(1 for v in st.session_state.schema.values() if v["type"] == "categorical")
            render_metric_card("Categorical Features", f"{cat_count}")
            
        st.subheader("Data Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Schema & Type Inference")
        schema_rows = []
        for col, info in st.session_state.schema.items():
            schema_rows.append({
                "Column": col,
                "Inferred Type": info["type"],
                "Missing Values": f"{info['missing_count']} ({info['missing_pct']}%)",
                "Unique Values": info["n_unique"],
                "Sample Values": str(info["sample_values"])
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

        st.subheader("Interactive Exploratory Data Analysis (EDA)")
        eda_col = st.selectbox("Select Feature for EDA Distribution:", list(st.session_state.schema.keys()), key="eda_select")
        if eda_col:
            col_type = st.session_state.schema[eda_col]["type"]
            fig_eda = plot_interactive_eda(df, eda_col, col_type)
            st.plotly_chart(fig_eda, use_container_width=True)

# TAB 2: SYNTHETIC GENERATOR
with tabs[1]:
    if st.session_state.real_df is None:
        st.info("👈 Please load a dataset first from the sidebar.")
    else:
        st.write(f"Configure parameters in sidebar and click below to synthesize using **{method.upper()}**.")
        if st.button("🚀 Train & Generate Synthetic Data", type="primary"):
            with st.spinner(f"Fitting {method.upper()} model and generating {n_generate} samples..."):
                try:
                    synth = TabularSynthesizer(method=method, random_state=int(seed), epochs=int(ctgan_epochs))
                    synth.fit(st.session_state.real_df)
                    st.session_state.synth_df = synth.sample(n_rows=int(n_generate))
                    
                    # Compute comprehensive evaluation metrics
                    st.session_state.eval_results = evaluate_datasets(
                        st.session_state.real_df,
                        st.session_state.synth_df,
                        st.session_state.schema
                    )
                    st.session_state.ml_results = None
                    st.success(f"Generated {len(st.session_state.synth_df)} synthetic rows successfully!")
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")

        if st.session_state.synth_df is not None:
            st.subheader(f"Generated Dataset Preview ({len(st.session_state.synth_df)} Rows)")
            st.dataframe(st.session_state.synth_df.head(10), use_container_width=True)

# TAB 3: QUALITY & STATISTICAL FIDELITY
with tabs[2]:
    if st.session_state.eval_results is None or st.session_state.synth_df is None:
        st.info("Generate synthetic data in the 'Synthetic Generator' tab to view statistical fidelity metrics.")
    else:
        eval_res = st.session_state.eval_results
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            render_metric_card("Statistical Fidelity", f"{eval_res['overall_quality']}%", "Average 1-KS / 1-TVD")
        with col_m2:
            render_metric_card("Correlation Similarity", f"{eval_res['correlation_similarity']}%", "Matrix L1 distance similarity")
        with col_m3:
            sd_q = eval_res.get("sdmetrics", {}).get("quality_score", "N/A")
            render_metric_card("SDMetrics Quality Score", f"{sd_q}%" if sd_q != "N/A" else "N/A", "SDMetrics Benchmark")
        with col_m4:
            sd_d = eval_res.get("sdmetrics", {}).get("diagnostic_score", "N/A")
            render_metric_card("SDMetrics Diagnostic", f"{sd_d}%" if sd_d != "N/A" else "N/A", "Structure & Coverage")

        # SDMetrics breakdown
        if eval_res.get("sdmetrics", {}).get("success"):
            st.subheader("SDMetrics Single-Table Properties Breakdown")
            props_df = pd.DataFrame(eval_res["sdmetrics"]["quality_properties"])
            if not props_df.empty:
                st.dataframe(props_df, use_container_width=True)

        st.subheader("Per-Feature Statistical Metrics")
        metrics_table = []
        for col_name, m in eval_res["column_metrics"].items():
            if m["type"] == "numerical":
                metrics_table.append({
                    "Feature": str(col_name),
                    "Type": "Numerical",
                    "Fidelity Score": f"{m.get('fidelity_score', 0.0)}%",
                    "KS Statistic": f"{m.get('ks_statistic', 0.0):.4f}",
                    "KS p-value": f"{m.get('ks_pvalue', 0.0):.4f}",
                    "Wasserstein Dist": f"{m.get('wasserstein_dist', 0.0):.4f}",
                    "Real Mean ± Std": f"{m.get('real_mean')} ± {m.get('real_std')}",
                    "Synth Mean ± Std": f"{m.get('synth_mean')} ± {m.get('synth_std')}",
                })
            else:
                metrics_table.append({
                    "Feature": str(col_name),
                    "Type": "Categorical",
                    "Fidelity Score": f"{m.get('fidelity_score', 0.0)}%",
                    "Total Variation Dist": f"{m.get('tvd', 0.0):.4f}",
                    "Unique Real": str(m.get("unique_real", "-")),
                    "Unique Synth": str(m.get("unique_synth", "-")),
                    "KS Statistic": "N/A",
                    "KS p-value": "N/A",
                    "Wasserstein Dist": "N/A"
                })
        st.dataframe(pd.DataFrame(metrics_table), use_container_width=True)

        st.subheader("Distribution Visualizations (Real vs Synthetic)")
        selected_col = st.selectbox("Inspect Feature Distribution:", list(st.session_state.schema.keys()))
        if selected_col:
            col_type = st.session_state.schema[selected_col]["type"]
            fig = plot_distributions_comparison(
                st.session_state.real_df,
                st.session_state.synth_df,
                selected_col,
                col_type
            )
            st.plotly_chart(fig, use_container_width=True)

        num_cols = [c for c, val in st.session_state.schema.items() if val["type"] == "numerical"]
        if len(num_cols) >= 2:
            st.subheader("Multivariate Correlation Preservation")
            corr_fig = plot_correlation_heatmaps(st.session_state.real_df, st.session_state.synth_df, num_cols)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)

# TAB 4: PRIVACY & NEAREST NEIGHBOR ANALYSIS
with tabs[3]:
    if st.session_state.eval_results is None or st.session_state.synth_df is None:
        st.info("Generate synthetic data in the 'Synthetic Generator' tab to view privacy metrics.")
    else:
        priv = st.session_state.eval_results.get("privacy", {})
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_metric_card("Privacy Risk Level", priv.get("risk_level", "LOW"), "Overall Leakage Risk")
        with p2:
            render_metric_card("Exact Duplicates", f"{priv.get('exact_duplicates', 0)}", f"{priv.get('duplicate_risk_pct', 0.0)}% of synthetic data")
        with p3:
            render_metric_card("Mean DCR (Distance)", f"{priv.get('mean_dcr', 'N/A')}", "Normalized Feature Space")
        with p4:
            render_metric_card("Min DCR (Distance)", f"{priv.get('min_dcr', 'N/A')}", "Distance to Closest Real Record")

        st.markdown("""
        **Privacy Analysis Guide**:
        - **Exact Duplicates**: Evaluates whether any synthetic sample identically matches an original ground-truth row.
        - **Distance to Closest Record (DCR)**: Nearest-neighbor Euclidean distance between synthetic records and the real dataset. Safe generators maintain sufficient distance from training points without memorization.
        """)

# TAB 5: ML UTILITY (TSTR)
with tabs[4]:
    if st.session_state.synth_df is None or st.session_state.real_df is None:
        st.info("Please generate synthetic data first to test ML utility.")
    else:
        st.subheader("Train-on-Synthetic, Test-on-Real (TSTR) Benchmark")
        st.markdown("Evaluates whether a machine learning model trained on **only synthetic data** achieves comparable predictive performance when deployed on **real unseen data**.")
        
        target_candidate = st.selectbox(
            "Select Target Feature to Predict:",
            list(st.session_state.schema.keys())
        )
        
        if st.button("🧪 Run TSTR Utility Evaluation", type="primary"):
            with st.spinner("Training models on real vs synthetic datasets..."):
                try:
                    ml_res = evaluate_ml_utility(
                        st.session_state.real_df,
                        st.session_state.synth_df,
                        target_col=target_candidate,
                        schema=st.session_state.schema
                    )
                    st.session_state.ml_results = ml_res
                    st.success("TSTR benchmark completed!")
                except Exception as e:
                    st.error(f"Error computing ML utility: {str(e)}")

        if st.session_state.ml_results:
            res = st.session_state.ml_results
            u1, u2, u3 = st.columns(3)
            with u1:
                render_metric_card(f"TRTR Performance ({res['metric_name']})", f"{res['trtr_score']}%", "Trained on Real Data")
            with u2:
                render_metric_card(f"TSTR Performance ({res['metric_name']})", f"{res['tstr_score']}%", "Trained on Synthetic Data")
            with u3:
                render_metric_card("ML Utility Retention", f"{res['utility_ratio']}%", "TSTR / TRTR Ratio")

# TAB 6: EXPORT DATA & REPORTS
with tabs[5]:
    if st.session_state.synth_df is None:
        st.info("No synthetic data to export yet. Please generate data first.")
    else:
        st.subheader("1. Download Synthetic Dataset")
        csv_buffer = io.StringIO()
        st.session_state.synth_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download Synthetic CSV",
            data=csv_data,
            file_name="synthetic_dataset.csv",
            mime="text/csv",
            type="primary"
        )

        st.subheader("2. Export Evaluation & Audit Report")
        if st.session_state.eval_results:
            report_data = {
                "summary": {
                    "method": method,
                    "rows_generated": len(st.session_state.synth_df),
                    "statistical_fidelity_pct": st.session_state.eval_results["overall_quality"],
                    "correlation_similarity_pct": st.session_state.eval_results["correlation_similarity"],
                    "sdmetrics_quality_score": st.session_state.eval_results.get("sdmetrics", {}).get("quality_score"),
                    "privacy_risk_level": st.session_state.eval_results.get("privacy", {}).get("risk_level")
                },
                "privacy_assessment": st.session_state.eval_results.get("privacy"),
                "sdmetrics_assessment": st.session_state.eval_results.get("sdmetrics"),
                "ml_utility": st.session_state.ml_results,
                "column_metrics": st.session_state.eval_results["column_metrics"]
            }
            json_report = json.dumps(report_data, indent=2)
            st.download_button(
                label="📄 Download Full Audit Report (JSON)",
                data=json_report,
                file_name="synthetic_audit_report.json",
                mime="application/json"
            )
