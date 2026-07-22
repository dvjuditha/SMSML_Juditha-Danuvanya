import os
import json
import mlflow
import dagshub
from dotenv import load_dotenv
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import joblib

# Load environment variables
load_dotenv()

DAGSHUB_USER = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")

if DAGSHUB_USER and DAGSHUB_REPO:
    print(f"Connecting to DagsHub: {DAGSHUB_USER}/{DAGSHUB_REPO}")
    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
    if dagshub_token:
        dagshub.auth.add_app_token(dagshub_token)
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
else:
    print("WARNING: DAGSHUB_USER or DAGSHUB_REPO not found in .env. Falling back to local MLflow.")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("Breast Cancer - Advanced Tuning")

DATA_DIR = os.path.join("breast_cancer_preprocessing")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print("Starting Hyperparameter Tuning...")

# Grid Search params
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

gbc = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Best Params: {best_params}")
print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")

with mlflow.start_run(run_name="GBM-Tuned-BreastCancer"):
    # MANUAL LOGGING (Kriteria Advanced)
    # Log Parameters
    mlflow.log_params(best_params)
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    
    # Log Metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall
    })
    
    # Log Model as artifact
    mlflow.sklearn.log_model(best_model, "model")
    
    # Log Data Shapes as artifact (custom artifact)
    with open("data_shapes.txt", "w") as f:
        f.write(f"X_train: {X_train.shape}\n")
        f.write(f"X_test: {X_test.shape}\n")
    mlflow.log_artifact("data_shapes.txt")
    
# --- LOCAL ARTIFACTS DUMP ---
import json
os.makedirs("tuning_artifacts", exist_ok=True)
report = {
    "accuracy": accuracy,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "precision": precision,
    "recall": recall,
    "best_params": best_params
}
with open("tuning_artifacts/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)
print("Saved local JSON report to tuning_artifacts/classification_report.json")

print("Advanced tuning complete and logged manually to MLflow!")
