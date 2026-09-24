import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "FullBath",
    "BedroomAbvGr",
    "YearBuilt",
    "GarageCars",
    "TotalBsmtSF"
]

ENGINEERED_COLUMNS = [
    "TotalSF",
    "Qual_x_TotalSF",
    "PropertyAge",
    "Qual_Squared",
    "Qual_Cubed",
    "AvgRoomSF",
    "BathsPerBed",
    "BasementRatio",
    "HasBasement",
    "HasGarage",
    "IsNew",
    "IsLuxury"
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw house attributes into synergistic real-estate domain features.
    Designed for seamless integration into Scikit-Learn pipelines.
    
    Includes calibrated property age calculation bridging dataset epoch (2010)
    with real-world inference years, plus non-linear luxury scaling and space ratios.
    """
    def __init__(self, current_year=2026, dataset_epoch=2010):
        self.current_year = current_year
        self.dataset_epoch = dataset_epoch

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Support both pandas DataFrame and numpy array inputs with strict column order
        if isinstance(X, pd.DataFrame):
            df = X[FEATURE_COLUMNS].copy()
        else:
            df = pd.DataFrame(X, columns=FEATURE_COLUMNS)

        # 1. Total usable square footage (living space + basement)
        df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]

        # 2. Quality multiplied by Total SF (captures premium price acceleration for large luxury homes)
        df["Qual_x_TotalSF"] = df["OverallQual"] * df["TotalSF"]

        # 3. Property Age at evaluation:
        # Ames training data was recorded through 2010 (epoch). For houses built <= 2010,
        # age is calibrated to dataset_epoch so brand new homes at sale time have age ~0.
        # For modern constructions (YearBuilt > 2010), age is relative to current_year.
        # This keeps the age distribution [0, 136] perfectly aligned between training and inference.
        is_modern = df["YearBuilt"] > self.dataset_epoch
        df["PropertyAge"] = np.where(
            is_modern,
            np.maximum(0, self.current_year - df["YearBuilt"]),
            np.maximum(0, self.dataset_epoch - df["YearBuilt"])
        )

        # 4. Non-linear quality scaling for luxury/economy differentiation
        df["Qual_Squared"] = df["OverallQual"] ** 2
        df["Qual_Cubed"] = (df["OverallQual"] / 10.0) ** 3

        # 5. Room & space ratios (clamped to avoid zero division)
        total_rooms = np.maximum(1, df["BedroomAbvGr"] + df["FullBath"])
        df["AvgRoomSF"] = df["GrLivArea"] / total_rooms
        df["BathsPerBed"] = df["FullBath"] / np.maximum(1, df["BedroomAbvGr"])
        df["BasementRatio"] = df["TotalBsmtSF"] / np.maximum(1, df["TotalSF"])

        # 6. Binary indicators for amenities and property tier
        df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(float)
        df["HasGarage"] = (df["GarageCars"] > 0).astype(float)
        df["IsNew"] = (df["PropertyAge"] <= 5).astype(float)
        df["IsLuxury"] = ((df["OverallQual"] >= 8) & (df["GrLivArea"] >= 2000)).astype(float)

        return df.values

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_COLUMNS + ENGINEERED_COLUMNS)

