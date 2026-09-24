import os
import sys
import numpy as np
import pandas as pd
import joblib

# Ensure UTF-8 output even on standard Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure src can be imported cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import FeatureEngineer, FEATURE_COLUMNS

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train.csv")
df = pd.read_csv(data_path)
initial_count = len(df)

# 2. Filter Ames dataset known distressed partial-sale outliers (>4000 sq ft, sold < $300k)
# Reference: Dean De Cock (Ames Housing Dataset documentation)
outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
df = df[~outlier_mask].copy()
print(f"[*] Dataset cleaned: removed {initial_count - len(df)} abnormal distressed sale outlier(s).")

# 3. Define web user input features
X = df[FEATURE_COLUMNS]
y = df["SalePrice"]

# 4. Train-test split for held-out evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Build robust ensemble pipeline with domain feature engineering and imputer
base_regressor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_engineer", FeatureEngineer()),
        ("scaler", RobustScaler()),
        ("model", VotingRegressor(
            estimators=[
                ("gb", GradientBoostingRegressor(
                    n_estimators=350,
                    learning_rate=0.035,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42
                )),
                ("hgb", HistGradientBoostingRegressor(
                    max_iter=350,
                    learning_rate=0.035,
                    max_depth=5,
                    min_samples_leaf=12,
                    random_state=42
                )),
                ("rf", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1
                )),
                ("ridge", Ridge(alpha=15.0))
            ],
            weights=[3, 3, 2, 1]
        ))
    ]
)

# Wrap in TransformedTargetRegressor to guarantee strictly positive predictions and stabilize variance
pipeline = TransformedTargetRegressor(
    regressor=base_regressor,
    func=np.log1p,
    inverse_func=np.expm1
)

# 6. Comprehensive Cross-Validation Evaluation (5-Fold)
print("[*] Running 5-fold cross-validation on full dataset...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
cv_mae = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")

print("-" * 50)
print("  5-Fold Cross-Validation Metrics:")
print(f"  - Mean R² Score: {cv_r2.mean():.4f} (std: {cv_r2.std():.4f})")
print(f"  - Mean MAE:      ${cv_mae.mean():,.2f} (std: ${cv_mae.std():,.2f})")
print("-" * 50)

# 7. Evaluate on held-out test split
print("[*] Training and evaluating on held-out test split (80/20)...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 50)
print("  Held-Out Test Set Performance Metrics:")
print(f"  - MAE:      ${mae:,.2f}")
print(f"  - RMSE:     ${rmse:,.2f}")
print(f"  - R² Score: {r2:.4f}")
print("-" * 50)

# 8. Retrain on complete clean dataset for production deployment
print("[*] Retraining final model on all clean training data...")
pipeline.fit(X, y)

# 9. Save model
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(model_dir, exist_ok=True)
save_path = os.path.join(model_dir, "house_price_web_model.pkl")

joblib.dump(pipeline, save_path)
print(f"[OK] Website prediction model saved successfully to: {save_path}")