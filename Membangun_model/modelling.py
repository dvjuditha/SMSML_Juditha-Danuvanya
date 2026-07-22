import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Breast Cancer - Basic")

DATA_DIR = os.path.join("breast_cancer_preprocessing")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape},  y_test : {y_test.shape}")

# Autolog untuk kriteria basic
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest-Basic"):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"\nModel trained. Test Accuracy: {score:.4f}")
    print("Artifacts saved to MLflow Tracking UI at http://127.0.0.1:5000")
