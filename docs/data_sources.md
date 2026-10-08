# Data Sources & Storage Plan

**Status:** v0.2 — storage layer migrated from PostgreSQL to BigQuery (2026-08-22)
**Roadmap step:** 3 — Warehouse setup & load
**Source of truth for decisions below:** the project design (see `methodology.md`)

> **Changelog v0.1 → v0.2:** PostgreSQL `raw`/`staging` schemas replaced by BigQuery
> datasets. §5.2's open decision (who owns `staging`) is now **closed: dbt owns it.**
> Series tables in §2–§4 updated to the 21 codes actually pulled.

---

## 1. Purpose

Two things this document nails down:

1. **Exactly which series we pull** — code, meaning, expected history.
2. **Where each stage of the data lives** — local landing files vs. BigQuery, and how the datasets divide responsibility with dbt.

Everything here maps 1:1 onto the folder structure in place (`data/raw`, `data/external`, `src/data_collection`).

---

## 2. Target series — BCRP "Pagos inmediatos con alias"

Endpoint family: `estadisticas.bcrp.gob.pe` (free, no auth).

**Implementation note (supersedes v0.1's "batch these 8 into a single call"):** the API accepts up to 10 codes per request, but does **not** return series in request order, and identifies each series only by a long `name` string — never by code. Verified against a live 10-code call where position 0 held the *last*-listed series. The ETL therefore requests **one code per call**, making the code→values mapping correct by construction. Cost: 21 requests instead of 3.

| Code | Series | `col_name` |
|---|---|---|
| PN42672EM | Número — Intrabancarias — Yape | `n_transf_intra_yape` |
| PN42673EM | Número — Intrabancarias — Plin | `n_transf_intra_plin` |
| PN42677EM | Número — Interbancarias (Visa Direct) — Yape | `n_transf_inter_visa_yape` |
| PN42678EM | Número — Interbancarias (Visa Direct) — Plin | `n_transf_inter_visa_plin` |
| PN42662EM | Valor (S/) — Intrabancarias — Yape | `v_transf_intra_yape` |
| PN42663EM | Valor (S/) — Intrabancarias — Plin | `v_transf_intra_plin` |
| PN42667EM | Valor (S/) — Interbancarias — Yape | `v_transf_inter_visa_yape` |
| PN42668EM | Valor (S/) — Interbancarias — Plin | `v_transf_inter_visa_plin` |

Jan 2024 – present, ~29 monthly observations. This is why Phase 1 modeling leans on classical methods with proper backtesting rather than anything data-hungry.

Naming convention: `n_` = número (count of operations), `v_` = valor/monto (S/).

## 3. Macro features (confirmed core feature set)

| Indicator | Code | `col_name` |
|---|---|---|
| Inflación (IPC) | PN38705PM | `ipc` |
| Tipo de cambio nominal promedio (S/ por US$) | PN01246PM | `tipo_cambio` |
| Tasa de Referencia de la Política Monetaria | PD04722MM | `tasa_referencia` |
| PBI mensual (índice 2007=100) | PN01770AM | `pbi_idx` |
| Coeficiente de Dolarización de la Liquidez (%) | PN00025MM | `dolarizacion_liquidez` |
| Circulante — Emisión Primaria MN (millones S/) | PN00048MM | `circulante` |
| Ingreso promedio sector formal privado — Nominal (S/) | PN37696PM | `ingreso_formal` |
| Empleo | **TBD — open item, see §7** | — |

## 4. Complementary — CCE aggregates

| Code | Series | `col_name` |
|---|---|---|
| PN42230EM | CCE — Cheques — Número (miles) | `n_cce_cheques` |
| PN42231EM | CCE — Transferencias de Crédito — Monto (millones S/) | `v_cce_credito` |
| PN42232EM | CCE — Transferencias de Crédito — Número (miles) | `n_cce_credito` |
| PN42233EM | CCE — Transferencias Inmediatas — Monto (millones S/) | `v_cce_inmediatas` |
| PN42234EM | CCE — Transferencias Inmediatas — Número (miles) | `n_cce_inmediatas` |
| PN42661EM | Pagos con alias — Valor Intrabancarias total | `v_alias_intra_tot` |

Confirmed active from 2024 onwards. **Open item:** the pre-2024 long-history CCE codes (back to 2017) that originally motivated including these are not yet located — see §7.

Optional, not yet implemented: Google Trends "Yape"/"Plin" via `pytrends`, as a higher-frequency adoption proxy.

---

## 5. Storage architecture

### 5.1 Local raw landing — `data/raw/`

Every API pull is written to disk **before** anything touches the warehouse, untouched, as an immutable snapshot. This is what makes the pipeline reproducible and debuggable without re-hitting the API — and it means **BigQuery can always be rebuilt from local files**. Local storage and the warehouse are not competing; the files are the landing zone, BigQuery is the query layer.

```
data/raw/
├── bcrp/
│   └── {pull_date}_{series_code}.json      # e.g. 2026-08-21_PN42672EM.json
└── google_trends/
    └── {pull_date}_yape_plin.csv           # not yet implemented
```

One file per series — so `source_batch` in the warehouse points straight back at the file that produced each row. `data/external/` stays reserved for anything not pulled by our own ETL (e.g. a manually-downloaded INEI file, once §7 is resolved).

### 5.2 BigQuery — three datasets, two owners

```
<project>.raw        ← loaded by Python ETL. One script. Dumb and literal.
<project>.staging    ← built by dbt. Never touched by hand.
<project>.marts      ← built by dbt. Analysis-ready.
```

**Decision (was open in v0.1, now closed):** the ETL only ever writes to `raw`. dbt reads `raw` as a *source* and builds everything above it. This is the standard dbt convention, and it's the only way to actually exercise dbt as a skill rather than splitting transformation logic between Python and SQL.

**Location:** pick one region (e.g. `southamerica-west1` for Lima, or `US` for the widest free-tier compatibility) and use it for **all three datasets**. BigQuery cannot join across locations — a mismatch here is painful to undo later.

### 5.3 Raw table design — long/tall, not one column per series

Rather than one column per series (which breaks every time a series is added), one long table for all BCRP pulls:

```sql
CREATE SCHEMA IF NOT EXISTS `<project>.raw`
  OPTIONS (location = 'southamerica-west1');

CREATE TABLE IF NOT EXISTS `<project>.raw.bcrp_observations` (
    series_code   STRING    NOT NULL,
    obs_date      DATE      NOT NULL,
    value         NUMERIC,
    pulled_at     TIMESTAMP NOT NULL,
    source_batch  STRING    NOT NULL   -- filename in data/raw/bcrp/, for traceability
)
PARTITION BY obs_date
CLUSTER BY series_code;

CREATE TABLE IF NOT EXISTS `<project>.raw.series_metadata` (
    series_code   STRING NOT NULL,
    description   STRING NOT NULL,
    api_name      STRING,            -- what BCRP itself calls this code
    category      STRING NOT NULL,   -- 'target' | 'macro' | 'complementary'
    frequency     STRING NOT NULL,
    col_name      STRING NOT NULL,
    history_start DATE
);
```

**Differences from the PostgreSQL draft, and why:**

- **No `PRIMARY KEY`.** BigQuery does not enforce constraints. Uniqueness is asserted by **dbt tests** on the staging model instead — which is arguably better practice, since the assertion is versioned, visible, and runs on every build.
- **`PARTITION BY obs_date` + `CLUSTER BY series_code`** replace the Postgres index. At this data size neither matters for performance; they're in place because *demonstrating you know they exist* is the point. Worth a sentence in the README.
- **Append-only is preserved.** Re-pulls insert new rows rather than overwriting, keeping full pull history. dbt's staging layer de-duplicates down to "latest value per series/date" — the newest pull wins, because BCRP revises recent months.
- **`api_name` added** to `series_metadata`, captured by `bcrp_client.fetch_series()`, so we record what the provider actually calls each code.

`series_metadata` mirrors `src/data_collection/config.py` — **keep the two in sync.** The loader should write it from `ALL_SERIES` rather than by hand, so drift is impossible.

### 5.4 The load step

The only network write in the pipeline:

```python
from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID)
job = client.load_table_from_dataframe(
    df,                                   # long format: series_code, obs_date, value, ...
    f"{PROJECT_ID}.raw.bcrp_observations",
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
)
job.result()
```

At this scale (~600 rows), a direct load job is correct. The production-scale pattern would stage through a Cloud Storage bucket first — worth *mentioning* in the README to show the distinction is understood, not worth implementing here.

### 5.5 Naming conventions

- BCRP series codes stored exactly as returned by the API (`PN42672EM`), never renamed — `series_metadata` is where human-readable labels live.
- Landing filenames: `{ISO date}_{series_code}.json`.
- BigQuery datasets: `raw`, `staging` (dbt), `marts` (dbt).

---

## 6. Credentials & access

- Local development: `gcloud auth application-default login`. **No key file to leak.**
- `.env` holds non-secret config only — `GCP_PROJECT_ID`, `BQ_LOCATION`, `BQ_DATASET_RAW`.
- A service-account JSON key is needed **only** for GitHub Actions (step 11), and lives in GitHub Secrets — never in the repo.
- `.gitignore` must exclude `.env`, `.venv/`, `.venv-dbt/`, and `*.json` service-account keys.

---

## 7. Open items

1. **Empleo series.** BCRP's employment series look thinner than INEI's. Next step: search INEI's ENAHO / employment survey data for a monthly (or best-available-frequency) series that pairs with the BCRP macro set. Flagged as a research task rather than guessing at a code.
2. **Pre-2024 CCE long-history codes.** The CCE codes currently in `config.py` only reach back to 2024, which defeats their original purpose (extending effective history to 2017). The longer-history codes need to be located, or this rationale dropped from the outline.

---

## 8. Next steps

1. Create the GCP project and enable the BigQuery API.
2. Create the `raw`, `staging`, `marts` datasets in a single chosen location.
3. Write `src/data_collection/load_to_bigquery.py` — parse snapshots (reuse `parse_snapshot()` from the EDA notebook, promoted into `src/`), load `raw.bcrp_observations`, and write `raw.series_metadata` from `config.ALL_SERIES`.
4. Verify row counts in BigQuery against the local snapshot count.
5. Resolve §7 open items.
