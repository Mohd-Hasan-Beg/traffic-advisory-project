import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

print("Traffic Advisory Project Starting...")

df = pd.read_csv("Dataset/traffic volume.csv")
print(f"Dataset loaded: {df.shape}")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "traffic_volume"]).copy()

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["weekday"] = df["timestamp"].dt.weekday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

for col in ["location_id", "weather_condition", "signal_status", "accident_reported"]:
    df[col] = df[col].astype(str)

df = df.drop(columns=["timestamp"])

for col in ["temperature", "humidity", "avg_vehicle_speed", "vehicle_count_cars", "vehicle_count_trucks", "vehicle_count_bikes"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna(df.median(numeric_only=True))

q1 = df["traffic_volume"].quantile(0.33)
q2 = df["traffic_volume"].quantile(0.66)

def bucket(x):
    if x <= q1:
        return 0
    elif x <= q2:
        return 1
    else:
        return 2

df["traffic_level"] = df["traffic_volume"].apply(bucket)
df = df.drop(columns=["traffic_volume"])

df = pd.get_dummies(df, columns=["location_id", "weather_condition", "signal_status", "accident_reported"], drop_first=True)

X = df.drop("traffic_level", axis=1)
y = df["traffic_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42, n_estimators=500, n_jobs=-1)

param_grid = {
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
pred = best_model.predict(X_test)

acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, pred))

joblib.dump(best_model, "traffic_level_model.pkl")
joblib.dump(list(X.columns), "traffic_model_columns.pkl")
joblib.dump(sorted(df.filter(like="location_id_").columns.tolist()), "location_columns.pkl")
joblib.dump({"q1": float(q1), "q2": float(q2)}, "traffic_thresholds.pkl")

print("Model saved.")
print("Run: python -m streamlit run traffic_app.py")