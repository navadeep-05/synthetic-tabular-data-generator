import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Curated vibrant multi-color palette designed for high contrast in both Light & Dark modes
VIBRANT_PALETTE = [
    "#3B82F6",  # Royal Blue
    "#10B981",  # Emerald
    "#F59E0B",  # Amber/Orange
    "#8B5CF6",  # Purple
    "#EC4899",  # Hot Pink
    "#06B6D4",  # Cyan
    "#F97316",  # Bright Orange
    "#14B8A6",  # Teal
    "#6366F1",  # Indigo
    "#84CC16"   # Lime Green
]

def _apply_theme_to_figure(fig: go.Figure) -> go.Figure:
    """
    Apply transparent canvas and responsive grid/font colors
    so charts adapt automatically to both Light and Dark themes.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, system-ui, sans-serif",
            size=12
        ),
        xaxis=dict(
            gridcolor="rgba(128, 128, 128, 0.2)",
            zerolinecolor="rgba(128, 128, 128, 0.3)",
            linecolor="rgba(128, 128, 128, 0.3)"
        ),
        yaxis=dict(
            gridcolor="rgba(128, 128, 128, 0.2)",
            zerolinecolor="rgba(128, 128, 128, 0.3)",
            linecolor="rgba(128, 128, 128, 0.3)"
        )
    )
    return fig

def plot_interactive_eda(df: pd.DataFrame, col: str, col_type: str) -> go.Figure:
    """Generate high-contrast, multi-color interactive Plotly visualization for EDA."""
    if col_type == "numerical":
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        fig = px.histogram(
            series,
            x=col,
            marginal="box",
            nbins=35,
            color_discrete_sequence=["#3B82F6"],
            title=f"Distribution & Spread: <b>{col}</b>"
        )
        fig.update_layout(
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            height=380
        )
    else:
        # Categorical: Multi-color distinct palette per category
        clean_series = df[col].dropna()
        if clean_series.empty:
            clean_series = df[col].astype(str)
            
        counts = clean_series.astype(str).value_counts(sort=True, ascending=False).reset_index()
        counts.columns = [col, "Count"]
        top_counts = counts.head(15)
        
        if top_counts.empty:
            top_counts = pd.DataFrame({col: ["No Data"], "Count": [0]})
        
        fig = px.bar(
            top_counts,
            x=col,
            y="Count",
            text="Count",
            color=col,
            color_discrete_sequence=VIBRANT_PALETTE,
            title=f"Category Frequency: <b>{col}</b> (Top {len(top_counts)})"
        )
        fig.update_traces(
            textposition="outside",
            marker=dict(line=dict(width=1, color="rgba(0,0,0,0.15)"))
        )
        fig.update_layout(
            showlegend=False,
            xaxis_tickangle=-30,
            margin=dict(l=20, r=20, t=50, b=20),
            height=380
        )
        
    return _apply_theme_to_figure(fig)

def plot_distributions_comparison(real_df: pd.DataFrame, synth_df: pd.DataFrame, col: str, col_type: str) -> go.Figure:
    """Side-by-side or overlaid interactive distribution comparison (Plotly)."""
    fig = go.Figure()
    
    if col_type == "numerical":
        r_vals = pd.to_numeric(real_df[col], errors="coerce").dropna()
        s_vals = pd.to_numeric(synth_df[col], errors="coerce").dropna()
        
        fig.add_trace(go.Histogram(
            x=r_vals,
            name="Real Ground Truth",
            opacity=0.65,
            marker_color="#2563EB",  # Vibrant Cobalt Blue
            histnorm="probability density"
        ))
        fig.add_trace(go.Histogram(
            x=s_vals,
            name="Synthetic Generated",
            opacity=0.65,
            marker_color="#10B981",  # Vibrant Emerald Green
            histnorm="probability density"
        ))
        fig.update_layout(
            barmode="overlay",
            title=f"Interactive Density Overlay: <b>{col}</b>",
            xaxis_title=col,
            yaxis_title="Probability Density",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            height=380,
            margin=dict(l=20, r=20, t=50, b=20)
        )
    else:
        r_clean = real_df[col].dropna()
        if r_clean.empty:
            r_clean = real_df[col].astype(str)
        r_dist = r_clean.astype(str).value_counts(normalize=True).reset_index()
        r_dist.columns = [col, "Proportion"]
        r_dist["Dataset"] = "Real"
        
        s_clean = synth_df[col].dropna()
        if s_clean.empty:
            s_clean = synth_df[col].astype(str)
        s_dist = s_clean.astype(str).value_counts(normalize=True).reset_index()
        s_dist.columns = [col, "Proportion"]
        s_dist["Dataset"] = "Synthetic"
        
        combined = pd.concat([r_dist, s_dist], ignore_index=True)
        top_cats = r_dist.nlargest(10, "Proportion")[col].tolist() if not r_dist.empty else []
        if top_cats:
            combined = combined[combined[col].isin(top_cats)]
        
        fig = px.bar(
            combined,
            x=col,
            y="Proportion",
            color="Dataset",
            barmode="group",
            color_discrete_map={"Real": "#2563EB", "Synthetic": "#10B981"},
            title=f"Relative Proportion Comparison: <b>{col}</b>"
        )
        fig.update_layout(
            xaxis_tickangle=-30,
            yaxis_tickformat=".1%",
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            height=380,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
    return _apply_theme_to_figure(fig)

def plot_correlation_heatmaps(real_df: pd.DataFrame, synth_df: pd.DataFrame, num_cols: list) -> go.Figure:
    """Side-by-side interactive correlation heatmaps with hover values."""
    if len(num_cols) < 2:
        return None
        
    r_corr = real_df[num_cols].corr().round(2)
    s_corr = synth_df[num_cols].corr().round(2)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Real Data Correlation", "Synthetic Data Correlation"),
        horizontal_spacing=0.12
    )
    
    fig.add_trace(
        go.Heatmap(
            z=r_corr.values,
            x=num_cols,
            y=num_cols,
            zmin=-1, zmax=1,
            colorscale="RdBu_r",
            showscale=False,
            text=r_corr.values,
            texttemplate="%{text}"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=s_corr.values,
            x=num_cols,
            y=num_cols,
            zmin=-1, zmax=1,
            colorscale="RdBu_r",
            text=s_corr.values,
            texttemplate="%{text}"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="<b>Multivariate Correlation Matrix Comparison</b>",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return _apply_theme_to_figure(fig)
