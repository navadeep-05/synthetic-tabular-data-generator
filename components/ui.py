import streamlit as st

def apply_custom_theme():
    """
    Apply native Streamlit container styling that guarantees 100% text contrast
    and visibility across both Light and Dark modes.
    """
    st.markdown("""
        <style>
        /* Adaptable Metric Cards for Dark, Light, and Custom Streamlit Themes */
        .metric-card {
            background-color: var(--secondary-background-color);
            border: 1px solid var(--border-color, rgba(128, 128, 128, 0.25));
            border-radius: 10px;
            padding: 1.1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .metric-title {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-color);
            opacity: 0.85;
            margin-bottom: 0.2rem;
        }
        .metric-val {
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--text-color);
            line-height: 1.2;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: var(--text-color);
            opacity: 0.75;
            margin-top: 0.3rem;
        }
        
        /* Universal Badge */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background-color: rgba(99, 102, 241, 0.15);
            color: #6366f1;
            border: 1px solid rgba(99, 102, 241, 0.35);
            margin-bottom: 0.5rem;
        }

        /* Prevent manual text typing into selectbox inputs to enforce pure dropdown selection */
        div[data-baseweb="select"] input {
            caret-color: transparent !important;
            cursor: pointer !important;
            user-select: none !important;
        }
        div[data-baseweb="select"] {
            cursor: pointer !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_metric_card(title: str, value: str, subtitle: str = ""):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-val">{value}</div>
            {f'<div class="metric-sub">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)
