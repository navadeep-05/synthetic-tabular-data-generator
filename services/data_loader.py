import io
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

def load_data(file_buffer) -> pd.DataFrame:
    """Load CSV or Excel file buffer into a DataFrame with validation."""
    if file_buffer is None:
        raise ValueError("No file provided.")
    
    file_name = getattr(file_buffer, "name", "").lower()
    try:
        if file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_buffer)
        else:
            df = pd.read_csv(file_buffer)
    except Exception as e:
        raise ValueError(f"Failed to parse file: {str(e)}")
        
    if df.empty:
        raise ValueError("Uploaded file contains an empty dataset.")
    if len(df.columns) == 0:
        raise ValueError("Dataset has no columns.")
    return df

def analyze_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Infer column schema, missing values, unique counts, and suggested types."""
    schema = {}
    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_pct = float(missing_count / len(df) * 100) if len(df) > 0 else 0.0
        n_unique = int(s.nunique(dropna=True))
        
        # Determine inferred type
        if pd.api.types.is_numeric_dtype(s):
            col_type = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(s):
            col_type = "datetime"
        elif pd.api.types.is_bool_dtype(s):
            col_type = "categorical"
        else:
            col_type = "categorical"
            
        schema[str(col)] = {
            "type": col_type,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 2),
            "n_unique": n_unique,
            "sample_values": s.dropna().head(3).tolist(),
        }
    return schema
