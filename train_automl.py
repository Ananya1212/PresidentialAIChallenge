import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("safeflow_ai_simulated_dataset.csv")

# ----------------------------
# Target & features
# ----------------------------
TARGET = "average_speed"

# Drop columns that leak target or are IDs
DROP_COLS = [
    "average_speed",
    "traffic_volume",
    "congestion_level",
    "accident_risk",
    "road_id",
    "start_node",
    "end_node",
    "neighborhood_id"
]

X = df.drop(columns=DROP_COLS)
y = df[TARGET]

# One-hot encode categorical variables
X = pd.get_dummies(X)

# ----------------------------
# Train / test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# AutoML
# ----------------------------
automl = AutoML()

automl.fit(
    X_train,
    y_train,
    task="regression",
    time_budget=120,        # 2 minutes is enough
    metric="rmse",
    estimator_list=["lgbm", "rf", "xgboost"],
    seed=42
)

# ----------------------------
# Evaluation
# ----------------------------
preds = automl.predict(X_test)

rmse = ((preds - y_test) ** 2).mean() ** 0.5
print(f"Test RMSE: {rmse:.2f}")

# ----------------------------
# Save model (CORRECT WAY)
# ----------------------------
import pickle

with open("safeflow_speed_model.pkl", "wb") as f:
    pickle.dump(automl, f)

print("Model trained and saved!")

