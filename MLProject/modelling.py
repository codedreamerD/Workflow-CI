import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mlflow.set_experiment("Student_performance_CI")

data = pd.read_csv("dataset_preprocessed/students_performance_preprocessed.csv")

# Split fitur dan target
X = data.drop(['math score', 'reading score', 'writing score', 'average_score', 'performance_level'], axis=1)
y = data['performance_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

input_example = X_train.iloc[0:5]

mlflow.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    output_path = "models/best_model.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")