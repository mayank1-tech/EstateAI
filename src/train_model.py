import os
import sys
import pandas as pd
import numpy as np

# Ensure UTF-8 output even on standard Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train.csv")
df = pd.read_csv(data_path)

# Remove ID column
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Separate features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Numerical preprocessing with scaling for stability
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Define models with log target transformation
models = {
    "Ridge Regression": Ridge(alpha=10.0),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=250,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.85,
        random_state=42
    )
}

results = {}

print("\n[*] Training Machine Learning Models on All 79 Features...\n")

# Train and evaluate each model
for name, model in models.items():

    # Create complete pipeline with log target transformation
    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    ttr_pipeline = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # Train model
    ttr_pipeline.fit(X_train, y_train)

    # Make predictions
    predictions = ttr_pipeline.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Save results
    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }

    print(f"Model: {name}")
    print(f"MAE:      ${mae:,.2f}")
    print(f"RMSE:     ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print("-" * 40)


# Display comparison
print("\n[=] MODEL PERFORMANCE COMPARISON\n")

results_df = pd.DataFrame(results).T
print(results_df.to_string())

# Find best model based on R2 Score
best_model_name = results_df["R2 Score"].idxmax()

print(f"\n[+] Best Model: {best_model_name}")
print("[OK] Model training completed successfully!")