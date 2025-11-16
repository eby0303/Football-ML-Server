import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer

# ---- LOAD MODEL ----
pipeline = joblib.load("cat_boost_new.pkl")

# ---- LOAD SAMPLE INPUT (first row from your prediction CSV) ----
input_df = pd.read_csv("barca_final_25_26.csv").iloc[[0]]

# ---- EXTRACT PREPROCESSOR ----
preprocessor: ColumnTransformer = pipeline.named_steps["preprocessing"]

# ---- GET FEATURE NAMES AFTER PREPROCESSING ----
feature_names = preprocessor.get_feature_names_out()

print("\nTotal model input features:", len(feature_names))

# ---- RUN PREDICTION ----
prediction = pipeline.predict(input_df)[0]   # 1 row → output shape (21 targets)

# ---- GET TARGET COLUMN NAMES ----
target_columns = [
    "For_GF", "For_GA", "For_Result", "Expected_xG", "Expected_npxG",
    "Standard_Gls", "Standard_Sh", "Standard_SoT", "Standard_G/Sh", "Standard_G/SoT",
    "Performance_PSxG",
    "Tackles_Tkl", "Tackles_TklW", "Challenges_Tkl", "Challenges_Tkl%",
    "Total_PrgDist", "Total_TotDist", "Carries_PrgDist",
    "Carries_Carries", "Touches_Touches",
    "Total_Cmp%"
]

print("\nTotal target outputs:", len(target_columns))

# ---- CREATE A NICE DATAFRAME: target → predicted_value ----
result_df = pd.DataFrame({
    "Target": target_columns,
    "Predicted Value": prediction
})

print("\n===== PREDICTIONS =====")
print(result_df)

# ---- SAVE TO CSV ----
result_df.to_csv("prediction_output.csv", index=False)
print("\n✅ Saved: prediction_output.csv")

# ---- ALSO PRINT INPUT FEATURES WITH VALUES ----
input_values_df = pd.DataFrame({
    "Feature Name": input_df.columns,
    "Value": input_df.iloc[0].values
})

print("\n===== INPUT FEATURE VALUES =====")
print(input_values_df)

input_values_df.to_csv("input_features_used.csv", index=False)
print("\n✅ Saved: input_features_used.csv")
