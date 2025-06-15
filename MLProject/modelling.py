import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

# Set the experiment name
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

with mlflow.start_run():
    # Set parameter model
    n_estimators = 500
    max_depth = 20

    # model Random Forest
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Prediksi untuk evaluasi
    y_pred = model.predict(X_test)

    # Hitungan metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log parameter manual
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log metrik manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Simpan model dengan input_example
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )


    print("Model training dan logging selesai.")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")

output_path = "./models/best_model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump(model, output_path)
print(f"Model saved to: {output_path}")
