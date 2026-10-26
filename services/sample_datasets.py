import pandas as pd
import numpy as np

def generate_sample_dataset(dataset_type: str = "customer_churn", n_rows: int = 200) -> pd.DataFrame:
    """Provides sample real-world datasets for immediate generation and testing."""
    np.random.seed(42)
    
    if dataset_type == "customer_churn":
        data = {
            "customer_id": [f"CUST-{1000 + i}" for i in range(n_rows)],
            "age": np.random.randint(18, 70, size=n_rows),
            "tenure_months": np.random.randint(1, 60, size=n_rows),
            "monthly_charges": np.round(np.random.uniform(20.0, 120.0, size=n_rows), 2),
            "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.25, 0.20], size=n_rows),
            "payment_method": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_rows),
            "churn": np.random.choice(["Yes", "No"], p=[0.27, 0.73], size=n_rows)
        }
        # Add correlation: higher monthly charge -> slightly higher churn probability
        df = pd.DataFrame(data)
        return df

    elif dataset_type == "medical_vitals":
        data = {
            "patient_id": [f"PAT-{5000 + i}" for i in range(n_rows)],
            "gender": np.random.choice(["Male", "Female", "Other"], p=[0.48, 0.48, 0.04], size=n_rows),
            "systolic_bp": np.random.normal(120, 15, size=n_rows).round(1),
            "diastolic_bp": np.random.normal(80, 10, size=n_rows).round(1),
            "heart_rate": np.random.randint(60, 100, size=n_rows),
            "cholesterol": np.random.normal(190, 30, size=n_rows).round(1),
            "smoker": np.random.choice(["Yes", "No"], p=[0.20, 0.80], size=n_rows)
        }
        return pd.DataFrame(data)

    else:
        # Financial transactions
        data = {
            "tx_id": [f"TXN-{10000 + i}" for i in range(n_rows)],
            "amount": np.round(np.random.exponential(scale=150.0, size=n_rows) + 5.0, 2),
            "category": np.random.choice(["Groceries", "Dining", "Electronics", "Utilities", "Travel"], size=n_rows),
            "device": np.random.choice(["Mobile", "Web", "POS"], p=[0.6, 0.3, 0.1], size=n_rows),
            "is_fraud": np.random.choice([0, 1], p=[0.97, 0.03], size=n_rows)
        }
        return pd.DataFrame(data)
