import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="House Price Analytics",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 AI-Powered House Price Analytics Dashboard")
st.write("Analyze house prices and important property features using data analytics.")

# Resolve dataset path dynamically relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

# Load dataset
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset file not found at: {DATA_PATH}")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


df = load_data()

# Sidebar
st.sidebar.title("📊 Dashboard Menu")

option = st.sidebar.radio(
    "Select Analysis",
    [
        "Dataset Overview",
        "House Price Analysis",
        "Feature Analysis",
        "Model Performance"
    ]
)


# ---------------- DATASET OVERVIEW ----------------

if option == "Dataset Overview":

    st.header("📋 Dataset Overview")

    if not df.empty:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Houses", f"{df.shape[0]:,}")
        col2.metric("Total Features", f"{df.shape[1]}")
        col3.metric(
            "Average House Price",
            f"${df['SalePrice'].mean():,.0f}"
        )

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Dataset Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(info_df, use_container_width=True)
    else:
        st.warning("Dataset is empty or could not be loaded.")


# ---------------- HOUSE PRICE ANALYSIS ----------------

elif option == "House Price Analysis":

    st.header("💰 House Price Analysis")

    if not df.empty:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Minimum Price",
            f"${df['SalePrice'].min():,.0f}"
        )

        col2.metric(
            "Average Price",
            f"${df['SalePrice'].mean():,.0f}"
        )

        col3.metric(
            "Maximum Price",
            f"${df['SalePrice'].max():,.0f}"
        )

        st.subheader("House Price Distribution")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(
            df["SalePrice"],
            bins=35,
            color="#4f46e5",
            edgecolor="#ffffff",
            alpha=0.85
        )

        ax.set_xlabel("Sale Price ($)", fontsize=11)
        ax.set_ylabel("Number of Houses", fontsize=11)
        ax.set_title("Distribution of House Prices", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Format x-axis with comma separation
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Dataset is empty.")


# ---------------- FEATURE ANALYSIS ----------------

elif option == "Feature Analysis":

    st.header("📈 Feature Analysis")

    if not df.empty:
        feature = st.selectbox(
            "Select a Feature to Analyze Against Sale Price",
            [
                "OverallQual",
                "GrLivArea",
                "TotalBsmtSF",
                "GarageCars",
                "YearBuilt"
            ]
        )

        st.subheader(f"{feature} vs Sale Price")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(
            df[feature],
            df["SalePrice"],
            alpha=0.6,
            color="#10b981",
            edgecolors="#065f46"
        )

        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel("Sale Price ($)", fontsize=11)
        ax.set_title(f"{feature} vs Sale Price", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Format y-axis with dollar currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"${y:,.0f}"))

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Dataset is empty.")


# ---------------- MODEL PERFORMANCE ----------------

elif option == "Model Performance":

    st.header("🤖 Machine Learning Model Performance")

    results = pd.DataFrame({
        "Model": [
            "Ridge Regression (Full 79 Features)",
            "Gradient Boosting (Full 79 Features)",
            "Random Forest (Full 79 Features)",
            "Decision Tree (Full 79 Features)",
            "Web Ensemble (7 Key Features + Engineering)"
        ],

        "MAE ($)": [
            16415.17,
            16390.06,
            17328.68,
            23340.05,
            18401.63
        ],

        "RMSE ($)": [
            25053.84,
            27905.85,
            29456.37,
            33421.85,
            26311.29
        ],

        "R² Score": [
            0.9182,
            0.8985,
            0.8869,
            0.8544,
            0.8747
        ]
    })

    st.dataframe(results, use_container_width=True)

    st.subheader("R² Score Comparison")

    st.bar_chart(
        results.set_index("Model")["R² Score"]
    )

    st.success(
        "🏆 Top Overall Model: Ridge Regression with Target Log-Transform (R²: 0.9182, MAE: $16,415.17) • Web Deployment Ensemble: R²: 0.8747 (held-out), MAE: $18,401.63"
    )