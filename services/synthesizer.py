import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

class TabularSynthesizer:
    """Production tabular data generator supporting Gaussian Copula, CTGAN, KDE, and GMM."""
    
    SUPPORTED_METHODS = ["copula", "ctgan", "kde", "gmm"]

    def __init__(self, method: str = "copula", random_state: Optional[int] = 42, epochs: int = 10):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method '{method}' not supported. Choose from {self.SUPPORTED_METHODS}")
        self.method = method
        self.random_state = random_state
        self.epochs = epochs
        self.fitted_ = False
        
        self.columns_: List[str] = []
        self.dtypes_: Dict[str, str] = {}
        self.cat_probs_: Dict[str, Dict[Any, float]] = {}
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.num_stats_: Dict[str, Dict[str, float]] = {}
        
        # Copula / Multivariate model params
        self.mean_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.gmm_model_: Optional[GaussianMixture] = None
        self.kde_models_: Dict[str, KernelDensity] = {}
        self.ctgan_model_: Optional[Any] = None

    def fit(self, df: pd.DataFrame, schema_override: Optional[Dict[str, str]] = None) -> "TabularSynthesizer":
        if df.empty or len(df) < 2:
            raise ValueError("Input DataFrame must have at least 2 rows for fitting.")
            
        self.columns_ = list(df.columns)
        self.dtypes_ = {}
        self.cat_probs_ = {}
        self.num_cols_ = []
        self.cat_cols_ = []
        self.num_stats_ = {}
        
        # Determine schema per column
        for col in self.columns_:
            col_type = schema_override.get(col) if schema_override else None
            if not col_type:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_type = "numerical"
                else:
                    col_type = "categorical"
            self.dtypes_[col] = col_type
            if col_type == "numerical":
                self.num_cols_.append(col)
            else:
                self.cat_cols_.append(col)

        if self.method == "ctgan":
            from ctgan import CTGAN
            # Convert categoricals to strings and fill missing values for CTGAN stability
            train_df = df.copy()
            for col in self.cat_cols_:
                train_df[col] = train_df[col].astype(str).fillna("__MISSING__")
            for col in self.num_cols_:
                train_df[col] = train_df[col].fillna(train_df[col].median() if not pd.isna(train_df[col].median()) else 0.0)
                self.num_stats_[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()) if len(df) > 1 else 1.0,
                    "is_integer": bool(pd.api.types.is_integer_dtype(df[col]))
                }

            ctgan = CTGAN(epochs=max(1, self.epochs), verbose=False)
            ctgan.fit(train_df, discrete_columns=self.cat_cols_)
            self.ctgan_model_ = ctgan

        else:
            # Fit Categorical Marginals
            for col in self.cat_cols_:
                s = df[col].astype(str).fillna("__MISSING__")
                val_counts = s.value_counts(normalize=True)
                self.cat_probs_[col] = {
                    "values": list(val_counts.index),
                    "probabilities": list(val_counts.values)
                }
                
            # Fit Numerical Marginals and Joint Correlation Structure
            if self.num_cols_:
                num_df = df[self.num_cols_].copy()
                for col in self.num_cols_:
                    median_val = num_df[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    num_df[col] = num_df[col].fillna(median_val)
                    self.num_stats_[col] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()) if len(df) > 1 else 1.0,
                        "is_integer": bool(pd.api.types.is_integer_dtype(df[col]))
                    }

                if self.method == "copula":
                    self.mean_ = num_df.mean().to_numpy()
                    cov = np.cov(num_df.to_numpy(), rowvar=False)
                    if cov.ndim == 0:
                        cov = np.array([[float(cov)]])
                    cov += np.eye(cov.shape[0]) * 1e-6
                    self.cov_ = cov

                elif self.method == "gmm":
                    n_components = min(5, len(num_df))
                    self.gmm_model_ = GaussianMixture(
                        n_components=n_components,
                        covariance_type="full",
                        random_state=self.random_state
                    )
                    self.gmm_model_.fit(num_df)

                elif self.method == "kde":
                    for col in self.num_cols_:
                        data = num_df[[col]].to_numpy()
                        std = np.std(data)
                        bw = 1.06 * (std if std > 1e-5 else 1.0) * (len(data) ** (-1/5))
                        kde = KernelDensity(kernel="gaussian", bandwidth=max(bw, 1e-3))
                        kde.fit(data)
                        self.kde_models_[col] = kde

        self.fitted_ = True
        return self

    def sample(self, n_rows: int = 100) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Synthesizer must be fitted before calling sample().")
        if n_rows <= 0:
            raise ValueError("n_rows must be greater than 0.")
            
        rng = np.random.RandomState(self.random_state)

        if self.method == "ctgan":
            synth_df = self.ctgan_model_.sample(n_rows)
            # Enforce numerical bounds and dtypes
            for col in self.num_cols_:
                stats = self.num_stats_[col]
                synth_df[col] = pd.to_numeric(synth_df[col], errors="coerce").fillna(stats["mean"])
                synth_df[col] = np.clip(synth_df[col], stats["min"], stats["max"])
                if stats["is_integer"]:
                    synth_df[col] = np.round(synth_df[col]).astype(int)
            return synth_df[self.columns_]

        synth_data: Dict[str, Any] = {}
        
        # Sample Categoricals and IDs
        for col in self.cat_cols_:
            # If column is an ID/Key column, generate distinct sequential/random synthetic IDs
            if col.lower().endswith("id") or col.lower().startswith("id"):
                prefix = col.upper().replace("_", "")[:3]
                synth_data[col] = [f"{prefix}-{rng.randint(10000, 99999)}" for _ in range(n_rows)]
            else:
                probs_info = self.cat_probs_[col]
                sampled_cat = rng.choice(
                    probs_info["values"],
                    size=n_rows,
                    p=probs_info["probabilities"]
                )
                synth_data[col] = sampled_cat
            
        # Sample Numericals
        if self.num_cols_:
            if self.method == "copula":
                sampled_matrix = rng.multivariate_normal(
                    mean=self.mean_,
                    cov=self.cov_,
                    size=n_rows
                )
                if sampled_matrix.ndim == 1:
                    sampled_matrix = sampled_matrix.reshape(-1, 1)
            elif self.method == "gmm":
                sampled_matrix, _ = self.gmm_model_.sample(n_rows)
            elif self.method == "kde":
                cols_data = []
                for col in self.num_cols_:
                    col_samples = self.kde_models_[col].sample(n_rows, random_state=self.random_state)
                    cols_data.append(col_samples.reshape(-1))
                sampled_matrix = np.column_stack(cols_data)
                
            for idx, col in enumerate(self.num_cols_):
                vals = sampled_matrix[:, idx]
                stats = self.num_stats_[col]
                vals = np.clip(vals, stats["min"], stats["max"])
                if stats["is_integer"]:
                    vals = np.round(vals).astype(int)
                synth_data[col] = vals

        synth_df = pd.DataFrame(synth_data)
        synth_df = synth_df[self.columns_]
        return synth_df
