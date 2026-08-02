# Stock Movement Predictor — MLOps Pipeline (OPPE1)

Predict whether an NSE stock's close price rises 5 minutes later, using an
end-to-end MLOps pipeline on GCP: **DVC** (data versioning), **Feast** (feature
store), **MLflow** (tracking + registry), and **CI + CML** on GitHub Actions.

**Roll:** 23F2004634 · **Term:** MAY 2026

## Problem
Binary classification on NSE minute data (open/high/low/close/volume, 2017–2021).
At each minute *t*, predict 1 if close at *t+5min* > close at *t*, else 0.
Features (from *t−10* to *t*): `rolling_avg_10` (mean close), `volume_sum_10`
(sum volume), plus `stock_name`. Data is NOT pre-sorted — the pipeline sorts by
timestamp. Two iterations: **v0** (AARTIIND, ABCAPITAL) then **merged v0+v1**
(adds ABFRL, ADANIENT, ADANIGAS).

## Files
| File | Purpose |
|------|---------|
| `data_processing.py` | Multi-stock feature engineering (`--nrows` for subset-first). Outputs `processed/train_<v>.csv`, `test_<v>.csv`. |
| `train_mlflow.py` | RF hyperparameter grid → logs to MLflow → registers best as `stock_direction`. Artifacts stored in GCS for portability. Run per version. |
| `evaluate.py` | CI: fetch best model from registry, score DVC-pulled test data, emit metrics + confusion plot. |
| `feature_repo/` | Feast offline store — entity `stock_name`, feature views for the rolling features; apply + materialize + point-in-time retrieval. |
| `tests/test_features.py` | Sanity test per feature (validated against raw close/volume). |
| `.github/workflows/ci.yml` | CI on main: WIF auth → dvc pull test data → sanity tests → fetch model → CML report on PR. |
| `requirements.txt` | Pinned versions. |

## Data sources
`StockAnalyticaData/v0` and `/v1` from `IITMBSMLOps/MLOPS_MAY_2026_OPPE1` (~90 MB).
Versioned with DVC; bytes in GCS, pointers in Git.

## GCP resources
- **Vertex AI Workbench** (e2-standard-4) — pipeline execution
- **Cloud Storage** `gs://23f2004634-oppe1` — DVC remote + MLflow artifact store
- **Workload Identity Federation** (`github-pool`) — keyless GitHub Actions → GCP

## Run (see RUNBOOK.md for the full commit-checkpointed sequence)
```bash
pip install -r requirements.txt
export MLFLOW_ARTIFACT_URI=gs://23f2004634-oppe1/mlflow-artifacts

# D2 data + DVC
python data_processing.py --data-dir StockAnalyticaData/v0 --version v0
dvc add processed/train_v0.csv processed/test_v0.csv && dvc push

# D3 Feast
cd feature_repo && feast apply && feast materialize 2017-01-01T00:00:00 2021-12-31T00:00:00 && cd ..

# D4/D5 train both iterations
python train_mlflow.py --version v0
python train_mlflow.py --version v1

# D6 CI runs on push/PR to main (see ci.yml)
```

## Notes
- Model accuracy is secondary per the brief; focus is the MLOps pipeline.
- Data lives in DVC/GCS, MLflow artifacts in GCS, only pointers + `mlflow.db`
  (registry metadata) in Git.
