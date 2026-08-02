"""
Deliverable 6 (CI) — fetch the best model from the MLflow registry, score the
DVC-pulled test set, emit metrics.json + confusion.png for the CML report.
Model artifacts live in GCS; CI authenticates via WIF so `models:/` resolves.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, mlflow.sklearn, pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

FEATURES = ["rolling_avg_10", "volume_sum_10"]
MODEL = os.getenv("MODEL_NAME", "stock_direction")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL}'")
version = os.getenv("MODEL_VERSION") or str(max(int(v.version) for v in versions))
print(f"Fetching {MODEL} v{version} from MLflow registry")
model = mlflow.sklearn.load_model(f"models:/{MODEL}/{version}")

df = pd.read_csv(os.getenv("TEST_CSV", "processed/test_v0.csv"))
pred = model.predict(df[FEATURES])
metrics = {
    "model_version": version,
    "accuracy": accuracy_score(df["target"], pred),
    "precision": precision_score(df["target"], pred, zero_division=0),
    "recall": recall_score(df["target"], pred, zero_division=0),
    "f1": f1_score(df["target"], pred, zero_division=0),
}
json.dump(metrics, open("metrics.json", "w"), indent=2)

ConfusionMatrixDisplay(confusion_matrix(df["target"], pred)).plot()
plt.title(f"Test confusion — acc={metrics['accuracy']:.3f}")
plt.savefig("confusion.png", bbox_inches="tight")
print("metrics:", metrics)
