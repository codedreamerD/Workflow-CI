import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Tracking ke MLflow lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Student_performance_CI")

# Load dataset
data = pd.read_csv("students_performance_preprocessed.csv")

# Split fitur dan target
X = data.drop(['math score', 'reading score', 'writing score', 'average_score', 'performance_level'], axis=1)
y = data['performance_level']

# Split data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Ambil contoh input untuk log model (harus DataFrame)
input_example = X_train.iloc[0:5]

# Run MLflow autolog
mlflow.autolog()

with mlflow.start_run():
    # Model Random Forest (parameter tetap, tidak tuning)
    model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    # Simpan model ke file (untuk upload ke GitHub)
    output_path = "models/best_model.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")