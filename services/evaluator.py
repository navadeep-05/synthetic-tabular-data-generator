import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split

def compute_column_metrics(real_s: pd.Series, synth_s: pd.Series, col_type: str) -> Dict[str, Any]:
    """Compute exact statistical metrics between real and synthetic column."""
    metrics = {"type": col_type}
    
    if col_type == "numerical":
        r_clean = pd.to_numeric(real_s, errors="coerce").dropna().to_numpy()
        s_clean = pd.to_numeric(synth_s, errors="coerce").dropna().to_numpy()
        
        if len(r_clean) > 0 and len(s_clean) > 0:
            ks_stat, p_val = ks_2samp(r_clean, s_clean)
            w_dist = wasserstein_distance(r_clean, s_clean)
            
            fidelity_score = max(0.0, 1.0 - float(ks_stat)) * 100.0
            
            metrics["ks_statistic"] = round(float(ks_stat), 4)
            metrics["ks_pvalue"] = round(float(p_val), 4)
            metrics["wasserstein_dist"] = round(float(w_dist), 4)
            metrics["real_mean"] = round(float(np.mean(r_clean)), 3)
            metrics["synth_mean"] = round(float(np.mean(s_clean)), 3)
            metrics["real_std"] = round(float(np.std(r_clean)), 3)
            metrics["synth_std"] = round(float(np.std(s_clean)), 3)
            metrics["fidelity_score"] = round(fidelity_score, 2)
        else:
            metrics["fidelity_score"] = 0.0

    else:
        # Categorical: Total Variation Distance (TVD)
        r_counts = real_s.astype(str).value_counts(normalize=True)
        s_counts = synth_s.astype(str).value_counts(normalize=True)
        
        all_cats = list(set(r_counts.index).union(set(s_counts.index)))
        tvd = 0.5 * sum(abs(r_counts.get(cat, 0.0) - s_counts.get(cat, 0.0)) for cat in all_cats)
        fidelity_score = max(0.0, 1.0 - tvd) * 100.0
        
        metrics["tvd"] = round(float(tvd), 4)
        metrics["unique_real"] = int(real_s.nunique())
        metrics["unique_synth"] = int(synth_s.nunique())
        metrics["fidelity_score"] = round(fidelity_score, 2)
        
    return metrics

def build_sdmetrics_metadata(df: pd.DataFrame, schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build SDMetrics single-table metadata dictionary from schema."""
    columns_meta = {}
    for col in df.columns:
        ctype = schema.get(col, {}).get("type", "categorical")
        if ctype == "numerical":
            columns_meta[col] = {"sdtype": "numerical"}
        elif ctype == "datetime":
            columns_meta[col] = {"sdtype": "datetime"}
        elif ctype == "boolean":
            columns_meta[col] = {"sdtype": "boolean"}
        else:
            columns_meta[col] = {"sdtype": "categorical"}
    return {"columns": columns_meta}

def run_sdmetrics_report(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Execute SDMetrics QualityReport and DiagnosticReport."""
    try:
        from sdmetrics.reports.single_table import QualityReport, DiagnosticReport

        metadata = build_sdmetrics_metadata(real_df, schema)
        
        # Quality report
        q_report = QualityReport()
        q_report.generate(real_df, synth_df, metadata, verbose=False)
        q_score = round(q_report.get_score() * 100, 2)
        q_properties = q_report.get_properties().to_dict(orient="records")
        
        # Diagnostic report
        d_report = DiagnosticReport()
        d_report.generate(real_df, synth_df, metadata, verbose=False)
        d_score = round(d_report.get_score() * 100, 2)
        d_properties = d_report.get_properties().to_dict(orient="records")

        return {
            "success": True,
            "quality_score": q_score,
            "diagnostic_score": d_score,
            "quality_properties": q_properties,
            "diagnostic_properties": d_properties
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def evaluate_privacy_risks(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate privacy risks:
    - Exact duplicate matching between synthetic and real rows.
    - Nearest-neighbor distance ratio (Distance to Closest Record / DCR).
    """
    # 1. Exact Duplicate Count
    # Align common columns
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    real_sub = real_df[common_cols].astype(str)
    synth_sub = synth_df[common_cols].astype(str)
    
    # Merge indicator
    merged = pd.merge(synth_sub, real_sub, how="inner")
    exact_duplicates_count = len(merged)
    duplicate_risk_pct = round((exact_duplicates_count / len(synth_df)) * 100, 2) if len(synth_df) > 0 else 0.0

    # 2. Nearest Neighbor Distance Analysis (DCR) on numerical columns
    num_cols = [c for c, val in schema.items() if val.get("type") == "numerical" and c in common_cols]
    dcr_metrics = {}
    
    if len(num_cols) >= 1:
        scaler = StandardScaler()
        r_num = real_df[num_cols].fillna(real_df[num_cols].median())
        s_num = synth_df[num_cols].fillna(real_df[num_cols].median())
        
        scaler.fit(r_num)
        r_scaled = scaler.transform(r_num)
        s_scaled = scaler.transform(s_num)
        
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(r_scaled)
        distances, _ = nn.kneighbors(s_scaled)
        
        mean_dcr = float(np.mean(distances))
        min_dcr = float(np.min(distances))
        p5_dcr = float(np.percentile(distances, 5))
        
        # Privacy risk indicator: if 5th percentile distance is near 0, risk is high
        risk_level = "LOW"
        if p5_dcr < 0.05 or duplicate_risk_pct > 1.0:
            risk_level = "HIGH"
        elif p5_dcr < 0.2:
            risk_level = "MEDIUM"

        dcr_metrics = {
            "mean_dcr": round(mean_dcr, 4),
            "min_dcr": round(min_dcr, 4),
            "p5_dcr": round(p5_dcr, 4),
            "risk_level": risk_level
        }
    else:
        dcr_metrics = {
            "mean_dcr": None,
            "min_dcr": None,
            "p5_dcr": None,
            "risk_level": "LOW" if duplicate_risk_pct < 1.0 else "MEDIUM"
        }

    return {
        "exact_duplicates": exact_duplicates_count,
        "duplicate_risk_pct": duplicate_risk_pct,
        **dcr_metrics
    }

def evaluate_ml_utility(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str,
    schema: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Train-on-Synthetic, Test-on-Real (TSTR) vs Train-on-Real, Test-on-Real (TRTR) ML Utility.
    """
    if target_col not in real_df.columns or target_col not in synth_df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
        
    # Exclude ID-like or target columns from features
    feature_cols = [
        c for c in real_df.columns 
        if c in synth_df.columns 
        and c != target_col 
        and not (c.lower().endswith("id") or c.lower().startswith("id"))
    ]
    if not feature_cols:
        feature_cols = [c for c in real_df.columns if c in synth_df.columns and c != target_col]
        
    if not feature_cols:
        raise ValueError("No feature columns available to predict target.")
        
    is_classification = schema.get(target_col, {}).get("type") == "categorical" or real_df[target_col].nunique() <= 10
    
    # Preprocess features into numerical format via dummy encoding
    all_combined = pd.concat([real_df[feature_cols], synth_df[feature_cols]], ignore_index=True)
    all_encoded = pd.get_dummies(all_combined, drop_first=True, dummy_na=False)
    
    X_real = all_encoded.iloc[:len(real_df)].fillna(0)
    X_synth = all_encoded.iloc[len(real_df):].fillna(0)
    
    # Split real dataset into train (TR) and test (TE)
    if is_classification:
        y_real = real_df[target_col].astype(str)
        y_synth = synth_df[target_col].astype(str)
        X_tr, X_te, y_tr, y_te = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
        
        # TRTR Model
        m_real = RandomForestClassifier(n_estimators=50, random_state=42)
        m_real.fit(X_tr, y_tr)
        preds_real = m_real.predict(X_te)
        trtr_score = accuracy_score(y_te, preds_real) * 100
        
        # TSTR Model (Train on synthetic, test on real test set)
        m_synth = RandomForestClassifier(n_estimators=50, random_state=42)
        m_synth.fit(X_synth, y_synth)
        preds_synth = m_synth.predict(X_te)
        tstr_score = accuracy_score(y_te, preds_synth) * 100
        metric_name = "Accuracy"
    else:
        # Regression
        y_real = pd.to_numeric(real_df[target_col], errors="coerce").fillna(0)
        y_synth = pd.to_numeric(synth_df[target_col], errors="coerce").fillna(0)
        X_tr, X_te, y_tr, y_te = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
        
        m_real = RandomForestRegressor(n_estimators=50, random_state=42)
        m_real.fit(X_tr, y_tr)
        preds_real = m_real.predict(X_te)
        
        m_synth = RandomForestRegressor(n_estimators=50, random_state=42)
        m_synth.fit(X_synth, y_synth)
        preds_synth = m_synth.predict(X_te)
        
        # Normalized accuracy based on Mean Absolute Error relative to target spread
        val_range = max(float(y_te.max() - y_te.min()), 1e-6)
        mae_real = float(np.mean(np.abs(y_te - preds_real)))
        mae_synth = float(np.mean(np.abs(y_te - preds_synth)))
        
        trtr_score = max(0.0, round((1.0 - (mae_real / val_range)) * 100.0, 2))
        tstr_score = max(0.0, round((1.0 - (mae_synth / val_range)) * 100.0, 2))
        metric_name = "Normalized Accuracy (1 - NMAE)"
        
    utility_ratio = round((tstr_score / trtr_score) * 100, 2) if trtr_score > 0 else 0.0

    return {
        "task_type": "Classification" if is_classification else "Regression",
        "metric_name": metric_name,
        "trtr_score": round(trtr_score, 2),
        "tstr_score": round(tstr_score, 2),
        "utility_ratio": min(100.0, utility_ratio),
        "target_col": target_col
    }

def evaluate_datasets(real_df: pd.DataFrame, synth_df: pd.DataFrame, schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall quality score, correlation fidelity, privacy indicators, SDMetrics and column metrics."""
    col_metrics = {}
    fidelity_scores = []
    
    for col in real_df.columns:
        if col in synth_df.columns:
            ctype = schema.get(col, {}).get("type", "categorical")
            m = compute_column_metrics(real_df[col], synth_df[col], ctype)
            col_metrics[col] = m
            if "fidelity_score" in m:
                fidelity_scores.append(m["fidelity_score"])
                
    overall_quality = round(float(np.mean(fidelity_scores)), 2) if fidelity_scores else 0.0
    
    num_cols = [c for c, val in schema.items() if val.get("type") == "numerical" and c in real_df.columns and c in synth_df.columns]
    corr_similarity = 100.0
    real_corr_dict = {}
    synth_corr_dict = {}
    
    if len(num_cols) >= 2:
        real_corr = real_df[num_cols].corr().fillna(0.0)
        synth_corr = synth_df[num_cols].corr().fillna(0.0)
        diff = np.abs(real_corr.to_numpy() - synth_corr.to_numpy())
        mean_diff = float(np.mean(diff))
        corr_similarity = max(0.0, round((1.0 - (mean_diff / 2.0)) * 100.0, 2))
        real_corr_dict = real_corr.round(3).to_dict()
        synth_corr_dict = synth_corr.round(3).to_dict()
        
    # Privacy analysis
    privacy = evaluate_privacy_risks(real_df, synth_df, schema)
    
    # SDMetrics Quality and Diagnostic reports
    sdmetrics_res = run_sdmetrics_report(real_df, synth_df, schema)

    return {
        "overall_quality": overall_quality,
        "correlation_similarity": corr_similarity,
        "column_metrics": col_metrics,
        "num_columns_evaluated": len(col_metrics),
        "real_corr": real_corr_dict,
        "synth_corr": synth_corr_dict,
        "privacy": privacy,
        "sdmetrics": sdmetrics_res
    }
