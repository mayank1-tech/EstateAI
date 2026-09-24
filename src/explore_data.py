import pandas as pd

# Load the training dataset
df = pd.read_csv("data/train.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])