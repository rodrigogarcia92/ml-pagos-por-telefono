# ML — Pagos por Teléfono

[![CI](https://github.com/rodrigogarcia92/ml-pagos-por-telefono/actions/workflows/ci.yml/badge.svg)](https://github.com/rodrigogarcia92/ml-pagos-por-telefono/actions/workflows/ci.yml)

Forecasting Peruvian phone-banking usage (**Yape** and **Plin**) from BCRP
central-bank data, built as a full pipeline: API ingestion → cloud warehouse →
dbt transformations → forecasting models → served API.

> Portfolio project, work in progress. Status and decisions live in
> [`docs/methodology.md`](docs/methodology.md).

## Status

| Layer | State |
|---|---|
| Ingestion (BCRP API → immutable JSON) | ✅ built · ⚠️ currently blocked: the API now serves a bot-protection challenge (training plan O-13) |
| Warehouse (BigQuery) + dbt (`raw → staging → marts`) | ✅ built, 19+ dbt tests |
| Modelling scaffold + MLflow + offline tests | ✅ built, 15 tests |
| Baselines and model grid | 🟡 started — 9 of ~116 planned runs; infrastructure test only |
| Panel / transfer-learning model (PyTorch) | planned |
| FastAPI → Docker → Cloud Run, Vertex AI demo, frontend | planned |

## Stack

| Layer | Tool |
|---|---|
| Ingestion | Python (`requests`) → immutable JSON snapshots on disk |
| Warehouse | Google BigQuery (`raw` / `staging` / `marts`) |
| Transformation | dbt (`dbt-bigquery`) |
| Modelling | scikit-learn, XGBoost, statsmodels → PyTorch panel model |
| Tracking | MLflow (SQLite backend, GCS artifact root) |
| Serving | FastAPI + Docker → Cloud Run |
| CI | GitHub Actions |

## Layout

```
data/raw/bcrp/         immutable API snapshots — the warehouse rebuilds from these
data/processed/        training snapshots (gitignored, rebuilt by snapshot.py)
docs/                  outline, training plan, runbook, MLflow setup, data sources
configs/               model, feature-set and sweep specifications (YAML)
notebooks/             EDA (VS Code # %% cells)
pagos_dbt/             dbt project
reports/               analysis write-ups
scripts/               MLflow server launcher, helpers
src/data_collection/   BCRP client, snapshot parser, BigQuery loader
src/model_training/    snapshot -> features -> folds -> metrics -> MLflow
tests/                 offline tests (synthetic panel; no network, no credentials)
```

## Setup

Two virtualenvs, never active at once: `.venv` for ETL and modelling, `.venv-dbt`
for dbt (its pins collide with MLflow's).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt

copy .env.example .env          # defaults are correct for this project

gcloud auth application-default login
gcloud auth application-default set-quota-project <your-project-id>
```

The full cold-start procedure is in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Pipeline

```powershell
python -m src.data_collection.fetch_target_series     # BCRP API -> data/raw/bcrp/
python -m src.data_collection.load_to_bigquery --dry-run
python -m src.data_collection.load_to_bigquery        # -> BigQuery raw dataset
# dbt build (in .venv-dbt), then:
python -m src.model_training.snapshot                 # warehouse -> parquet
python -m src.model_training.sweep --config configs/sweeps/s1_baselines.yaml --dry-run
```

The loader is append-only and idempotent: it skips snapshot files BigQuery
already holds. De-duplication to "latest value per series/month" happens in
dbt's staging layer, because BCRP revises recent months.

## Tests

```powershell
pytest
```

Offline by design: the suite builds a synthetic panel, so it needs no BigQuery,
no BCRP API and no credentials — which is also why CI can run it with no secrets.
It includes the shuffled-target leak canary ([`docs/training_plan.md`](docs/training_plan.md) §9.0).

## Design notes

- **22 BCRP series**, requested **one code per API call**. The API accepts ten,
  but returns them out of order and identifies each only by a long name string —
  never by code — so batching makes the code→values mapping unrecoverable.
- **~30 monthly observations** for the Yape/Plin split. This constraint drives
  the modelling approach: naive baselines and expanding-window backtesting first,
  a pre-registered evaluation protocol, and a panel model across longer related
  series rather than a naive univariate deep model.
- **Training is local by design**; the dataset is a few hundred kilobytes. Cloud
  appears for the warehouse, serving, and one Vertex AI demonstration job.
- **`data/raw/` is committed.** Local files are the reproducibility layer;
  BigQuery is the query layer. The warehouse can always be rebuilt from disk.
- **No credentials in the repository.** Authentication is Application Default
  Credentials; there is no service-account key file.

See [`docs/methodology.md`](docs/methodology.md),
[`docs/training_plan.md`](docs/training_plan.md) and
[`docs/data_sources.md`](docs/data_sources.md) for the design, the pre-registered
modelling protocol, and the warehouse decision record.
