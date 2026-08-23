# ML — Pagos por Teléfono

Forecasting Peruvian phone-banking usage (**Yape** and **Plin**) from BCRP
central-bank data, built as a full pipeline: API ingestion → cloud warehouse →
dbt transformations → forecasting models → served API.

> Portfolio project. Local repository, no public remote.

## Stack

| Layer | Tool |
|---|---|
| Ingestion | Python (`requests`) → immutable JSON snapshots on disk |
| Warehouse | Google BigQuery (`raw` / `staging` / `marts`) |
| Transformation | dbt (`dbt-bigquery`) |
| Modelling | statsmodels, XGBoost → PyTorch panel model |
| Tracking | MLflow |
| Serving | FastAPI + Docker → Cloud Run |

## Layout

```
data/raw/bcrp/        immutable API snapshots — the warehouse rebuilds from these
docs/                 outline, data sources & storage design
notebooks/            EDA (VS Code # %% cells)
src/data_collection/  BCRP client, snapshot parser, BigQuery loader
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt

copy .env.example .env          # then fill in GCP_PROJECT_ID

gcloud auth application-default login
gcloud auth application-default set-quota-project <your-project-id>
```

## Pipeline

```powershell
python -m src.data_collection.fetch_target_series     # BCRP API -> data/raw/bcrp/
python -m src.data_collection.load_to_bigquery --dry-run
python -m src.data_collection.load_to_bigquery        # -> BigQuery raw dataset
```

The loader is append-only and idempotent: it skips snapshot files BigQuery
already holds, so re-running it is a no-op rather than a duplicate load.
De-duplication to "latest value per series/month" happens in dbt's staging
layer, because BCRP revises recent months.

## Design notes

- **21 BCRP series**, requested **one code per API call**. The API accepts ten,
  but returns them out of order and identifies each only by a long name string
  — never by code — so batching makes the code→values mapping unrecoverable.
  One call per code is correct by construction.
- **~29 monthly observations** for the Yape/Plin split. This constraint drives
  the modelling approach: classical baselines with expanding-window backtesting
  first, and a panel model across many related series rather than a naive
  univariate deep model.
- **`data/raw/` is committed.** Local files are the reproducibility layer;
  BigQuery is the query layer. The warehouse can always be rebuilt from disk.

See [`docs/methodology.md`](docs/methodology.md) and
[`docs/data_sources.md`](docs/data_sources.md) for the full design and the
warehouse decision record.
