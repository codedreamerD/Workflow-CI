import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import dagshub

dagshub.init(repo_owner='codedreamerD', repo_name='SMSML_Fadhilah-Nurrahmayanti', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/codedreamerD/SMSML_Fadhilah-Nurrahmayanti.mlflow")
mlflow.set_experiment("Experiment Student Performance")

data = pd.read_csv("students_performance_preprocessed.csv")

X = data.drop(['math score', 'reading score', 'writing score', 'average_score', 'performance_level'], axis=1)
y = data['performance_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
input_example = X_train.iloc[0:5]

n_estimators_range = np.linspace(100, 1000, 5, dtype=int)
max_depth_range = np.linspace(5, 50, 5, dtype=int)

best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"Tuning_{n_estimators}_{max_depth}"):
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)

            # Manual logging
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("accuracy_manual", acc)

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                mlflow.sklearn.log_model(sk_model=model, artifact_path="best_model", input_example=input_example)

print("Tuning selesai.")
print(f"Model terbaik: {best_params}, Akurasi: {best_accuracy:.4f}")