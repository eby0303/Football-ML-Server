import joblib
import pandas as pd

# Load your model
model = joblib.load("cat_boost.pkl")

# Get the preprocessing step
preprocessor = model.named_steps['preprocessing']

# Get original column order
numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]
expected_order = numeric_features + categorical_features

print("=== EXACT COLUMN ORDER EXPECTED BY MODEL ===")
for i, col in enumerate(expected_order, 1):
    print(f"{i:2d}. {col}")

print(f"\nTotal columns: {len(expected_order)}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")