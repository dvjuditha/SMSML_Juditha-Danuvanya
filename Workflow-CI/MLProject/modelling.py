import os
import mlflow
import dagshub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# For CI/CD, these variables are injected by GitHub Actions secrets
DAGSHUB_USER = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")

if DAGSHUB_USER and DAGSHUB_REPO:
    print(f"Connecting to DagsHub: {DAGSHUB_USER}/{DAGSHUB_REPO}")
    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
    if dagshub_token:
        dagshub.auth.add_app_token(dagshub_token)
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)

mlflow.set_experiment("Breast Cancer - CI Pipeline")

DATA_DIR = "breast_cancer_preprocessing" # relative to MLProject execution path
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="CI-RandomForest"):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")
