import os
import pandas as pd
import joblib

from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ============================================================
# Paths (robust to where script is run from)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "safeflow_ai_simulated_dataset.csv"
)

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv(DATA_PATH)

print("Dataset loaded.")
print("Rows:", len(df))
print("Columns:", list(df.columns))

# ============================================================
# Feature columns (MATCHES YOUR DATASET)
# ============================================================
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_arrival_time",
    "is_dismissal_time",
    "weather_condition",
    "precipitation",
    "visibility_level",
    "num_lanes",
    "speed_limit",
    "distance_km",
    "is_intersection",
    "neighborhood_population",
    "working_population_pct",
    "students_population",
    "distance_to_school_m",
    "crosswalk_present",
    "crossing_guard_present"
]

TARGET_CONGESTION = "congestion_level"
TARGET_RISK = "accident_risk"

X = df[FEATURE_COLUMNS]
y_congestion = df[TARGET_CONGESTION]
y_risk = df[TARGET_RISK]

# ============================================================
# Preprocessing (One-hot encode categoricals)
# ============================================================
CATEGORICAL_COLS = [
    "weather_condition",
    "visibility_level"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
    ],
    remainder="passthrough"
)

X_processed = preprocessor.fit_transform(X)

# ============================================================
# Train / test split (same split for both targets)
# ============================================================
(
    X_train,
    X_test,
    y_cong_train,
    y_cong_test,
    y_risk_train,
    y_risk_test
) = train_test_split(
    X_processed,
    y_congestion,
    y_risk,
    test_size=0.2,
    random_state=42,
    stratify=y_congestion
)

# ============================================================
# AutoML — Congestion Model
# ============================================================
automl_congestion = AutoML()

automl_congestion.fit(
    X_train,
    y_cong_train,
    task="classification",
    time_budget=120,
    metric="accuracy",
    seed=42
)



# ============================================================
# AutoML — Accident Risk Model
# ============================================================
automl_risk = AutoML()

automl_risk.fit(
    X_train,
    y_risk_train,
    task="classification",
    time_budget=120,
    metric="accuracy",
    seed=42
)



# ============================================================
# Evaluation
# ============================================================
cong_acc = automl_congestion.score(X_test, y_cong_test)
risk_acc = automl_risk.score(X_test, y_risk_test)

print("\n=== MODEL PERFORMANCE ===")
print("Congestion accuracy:", round(cong_acc, 3))
print("Accident risk accuracy:", round(risk_acc, 3))
print("Best congestion model:", automl_congestion.model)
print("Best risk model:", automl_risk.model)

# ============================================================
# Save models + preprocessor
# ============================================================
joblib.dump(preprocessor, os.path.join(BASE_DIR, "preprocessor.pkl"))
joblib.dump(automl_congestion, os.path.join(BASE_DIR, "congestion_model.pkl"))
joblib.dump(automl_risk, os.path.join(BASE_DIR, "accident_model.pkl"))

print("\nSaved files:")
print(" - preprocessor.pkl")
print(" - congestion_model.pkl")
print(" - accident_model.pkl")