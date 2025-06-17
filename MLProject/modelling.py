import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Tracking ke MLflow lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Student_performance_CI")

# Load dataset
data = pd.read_csv("dataset_preprocessed/students_performance_preprocessed.csv")

# Split fitur dan target
X = data.drop(['math score', 'reading score', 'writing score', 'average_score', 'performance_level'], axis=1)
y = data['performance_level']

# Split data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Ambil contoh input untuk log model (harus DataFrame)
input_example = X_train.iloc[0:5]

# Jalankan autolog
mlflow.autolog()

# Jalankan training dan simpan model
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    # Simpan model secara manual untuk diupload
    output_path = "models/best_model.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")