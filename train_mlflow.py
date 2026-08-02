"""
Deliverables 4 & 5 — train + tune + track with MLflow, register best model.

Runs a hyperparameter grid, logs params/metrics/model per run, registers the
best run as `stock_direction`. MLflow artifacts are stored in GCS (set via
--artifact-uri / MLFLOW_ARTIFACT_URI) so CI can fetch the model portably.
Tracking metadata is in sqlite:///mlflow.db (committed to git for CI).

Usage:
    export MLFLOW_ARTIFACT_URI=gs://<your-bucket>/mlflow-artifacts
    python train_mlflow.py --version v0
    python train_mlflow.py --version v1
"""
import argparse, os
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = ["rolling_avg_10", "volume_sum_10"]
TARGET = "target"
REGISTERED_MODEL = "stock_direction"
EXPERIMENT = "stock_analytica"
GRID = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 8},
    {"n_estimators": 150, "max_depth": 12},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0")
    ap.add_argument("--train", default=None)
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    ap.add_argument("--artifact-uri", default=os.getenv("MLFLOW_ARTIFACT_URI"))  # gs://bucket/mlflow-artifacts
    args = ap.parse_args()
    train_path = args.train or f"processed/train_{args.version}.csv"

    mlflow.set_tracking_uri(args.tracking_uri)
    # create the experiment once, with a GCS artifact root so paths are portable for CI
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=args.artifact_uri)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_csv(train_path)
    X_tr, X_ev, y_tr, y_ev = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET])

    results = []
    for params in GRID:
        with mlflow.start_run(run_name=f"{args.version}_rf{params['n_estimators']}_d{params['max_depth']}") as run:
            model = RandomForestClassifier(random_state=1, n_jobs=-1, **params).fit(X_tr, y_tr)
            pred = model.predict(X_ev)
            metrics = {
                "accuracy": accuracy_score(y_ev, pred),
                "precision": precision_score(y_ev, pred, zero_division=0),
                "recall": recall_score(y_ev, pred, zero_division=0),
                "f1": f1_score(y_ev, pred, zero_division=0),
            }
            mlflow.log_params(params)
            mlflow.log_param("data_version", args.version)
            mlflow.log_param("n_train", len(X_tr))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")
            results.append((metrics["accuracy"], run.info.run_id, params))
            print(f"  {args.version} {params} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")

    results.sort(reverse=True)
    best_acc, best_id, best_params = results[0]
    mv = mlflow.register_model(f"runs:/{best_id}/model", REGISTERED_MODEL)
    print(f"\nBest: {best_params} acc={best_acc:.4f} -> registered {REGISTERED_MODEL} v{mv.version}")


if __name__ == "__main__":
    main()
