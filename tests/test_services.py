import pytest
import pandas as pd
import numpy as np
from services.data_loader import analyze_schema, load_data
from services.synthesizer import TabularSynthesizer
from services.evaluator import (
    compute_column_metrics,
    evaluate_datasets,
    evaluate_privacy_risks,
    evaluate_ml_utility,
    run_sdmetrics_report
)
from services.sample_datasets import generate_sample_dataset

def test_sample_datasets():
    df_churn = generate_sample_dataset("customer_churn", n_rows=100)
    assert len(df_churn) == 100
    assert "churn" in df_churn.columns
    assert "monthly_charges" in df_churn.columns

def test_analyze_schema():
    df = pd.DataFrame({
        "age": [25, 30, 45, 50],
        "city": ["NY", "SF", "LA", "NY"],
        "flag": [True, False, True, False]
    })
    schema = analyze_schema(df)
    assert schema["age"]["type"] == "numerical"
    assert schema["city"]["type"] == "categorical"
    assert schema["age"]["missing_count"] == 0

def test_synthesizer_copula():
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "label": ["A", "B", "A", "B", "A", "B"]
    })
    synth = TabularSynthesizer(method="copula", random_state=42)
    synth.fit(df)
    sampled = synth.sample(n_rows=20)
    assert len(sampled) == 20
    assert list(sampled.columns) == list(df.columns)
    assert set(sampled["label"].unique()).issubset({"A", "B"})

def test_synthesizer_gmm_kde():
    df = pd.DataFrame({
        "num": [10, 20, 30, 40, 50, 60, 70, 80],
        "cat": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"]
    })
    # Test GMM
    synth_gmm = TabularSynthesizer(method="gmm", random_state=42)
    synth_gmm.fit(df)
    gmm_samples = synth_gmm.sample(n_rows=15)
    assert len(gmm_samples) == 15

    # Test KDE
    synth_kde = TabularSynthesizer(method="kde", random_state=42)
    synth_kde.fit(df)
    kde_samples = synth_kde.sample(n_rows=15)
    assert len(kde_samples) == 15

def test_synthesizer_ctgan():
    df = pd.DataFrame({
        "feature_num": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "feature_cat": ["Alpha", "Beta", "Alpha", "Beta", "Alpha", "Beta"]
    })
    synth_ctgan = TabularSynthesizer(method="ctgan", random_state=42, epochs=2)
    synth_ctgan.fit(df)
    ctgan_samples = synth_ctgan.sample(n_rows=10)
    assert len(ctgan_samples) == 10
    assert list(ctgan_samples.columns) == ["feature_num", "feature_cat"]

def test_privacy_evaluation():
    df_real = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": ["A", "B", "C", "D", "E"]
    })
    df_synth = pd.DataFrame({
        "x": [1.05, 2.1, 3.05, 4.02, 5.01],
        "y": ["A", "B", "C", "D", "E"]
    })
    schema = analyze_schema(df_real)
    privacy_res = evaluate_privacy_risks(df_real, df_synth, schema)
    assert "exact_duplicates" in privacy_res
    assert "mean_dcr" in privacy_res
    assert privacy_res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

def test_ml_utility():
    df_real = pd.DataFrame({
        "f1": np.linspace(1, 100, 40),
        "f2": np.linspace(10, 200, 40),
        "target": ["Yes" if i % 2 == 0 else "No" for i in range(40)]
    })
    df_synth = df_real.copy()
    schema = analyze_schema(df_real)
    ml_res = evaluate_ml_utility(df_real, df_synth, target_col="target", schema=schema)
    assert ml_res["task_type"] == "Classification"
    assert "trtr_score" in ml_res
    assert "tstr_score" in ml_res
    assert ml_res["utility_ratio"] > 0

def test_sdmetrics_and_evaluator():
    df_real = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat": ["A", "A", "B", "B", "A"]
    })
    df_synth = pd.DataFrame({
        "num": [12.0, 19.0, 29.0, 41.0, 48.0],
        "cat": ["A", "B", "B", "B", "A"]
    })
    schema = analyze_schema(df_real)
    eval_res = evaluate_datasets(df_real, df_synth, schema)
    
    assert "overall_quality" in eval_res
    assert "sdmetrics" in eval_res
    assert "privacy" in eval_res
    assert eval_res["sdmetrics"]["success"] is True
    assert 0 <= eval_res["sdmetrics"]["quality_score"] <= 100
