# Production Synthetic Tabular Data Generator

A modular, production-ready synthetic tabular data generator built with Python and Streamlit.

## Features
- **Data Ingestion & Profiling**: Supports CSV, Excel files, and built-in benchmark datasets with automatic schema inference.
- **Multi-Method Synthesis**:
  - **Gaussian Copula**: Preserves multivariate correlations and continuous marginal distributions.
  - **Gaussian Mixture Models (GMM)**: Models multi-modal continuous distributions.
  - **Kernel Density Estimation (KDE)**: Non-parametric density sampling.
  - Categorical marginal distribution preservation.
- **Statistical Fidelity Metrics**:
  - Two-sample Kolmogorov-Smirnov test (statistic and p-value).
  - Wasserstein distance (Earth Mover's Distance).
  - Total Variation Distance (TVD) for categorical columns.
  - Multivariate correlation matrix preservation check.
- **Interactive Visualizations**: Density distribution plots and side-by-side correlation heatmaps.
- **Data Export**: Export high-fidelity synthetic datasets to CSV.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run tests:
   ```bash
   pytest tests/
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
