import os
import sys
import pandas as pd
import numpy as np
import joblib

# Ensure UTF-8 output even on standard Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train.csv")
df = pd.read_csv(data_path)

# Filter abnormal partial-sale outliers (>4000 sq ft, sold < $300k)
outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
df = df[~outlier_mask].copy()

# Remove ID column
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Separate features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

# Numerical preprocessing
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create the final model pipeline with log-transformed target
base_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            random_state=42
        ))
    ]
)

final_pipeline = TransformedTargetRegressor(
    regressor=base_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

# Train the final model on full dataset
print("[*] Training the final Gradient Boosting model on all 79 features...")
final_pipeline.fit(X, y)

# Save the complete pipeline
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(model_dir, exist_ok=True)
save_path = os.path.join(model_dir, "house_price_model.pkl")

joblib.dump(final_pipeline, save_path)

print("\n[OK] Model saved successfully!")
print(f"Location: {save_path}")