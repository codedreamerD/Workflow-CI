import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_experiment("Student_performance_CI")

# Load data
data = pd.read_csv("students_performance_preprocessed.csv")

# Fitur dan target
X = data.drop(['math score', 'reading score', 'writing score', 'average_score', 'performance_level'], axis=1)
y = data['performance_level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

input_example = X_train.iloc[:5]

with mlflow.start_run():
    mlflow.autolog()  # otomatis log param, metric, dan model

    model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

print("Training selesai dan sudah dilog secara otomatis.")