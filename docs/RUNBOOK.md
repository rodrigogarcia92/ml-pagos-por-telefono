# Runbook — this project "ML - Pagos por Teléfono"

**Machine:** the development machine (Windows 11, PowerShell)
**Project root:** `<repo root>`
**Companions:** `docs/mlflow_setup.md` (one-time setup) · `docs/training_plan.md` (what to run and why) · `docs/methodology.md` (architecture)

> **What this is.** Every command, in order, from a cold machine to a finished
> sweep. Written to be followed without thinking, because the thinking is in the
> other three documents.
>
> **Read the decision tree in §0 first.** Most sessions do NOT need the ETL.

---

## 0. Which parts do I actually need today?

```
Cold machine
     |
     +-- Always:            §1  environment
     |
     +-- New BCRP data?  ---+-- yes --> §2 ETL   +  §3 dbt  +  §4 snapshot
     |                      |
     |                      +-- no  --> skip all three
     |
     +-- Changed config.py, or a dbt model? --> §3 dbt + §4 snapshot
     |
     +-- Only modelling code changed?        --> §5 onward
     |
     +-- Always, to train or look at runs:   §5 MLflow server
```

**BCRP publishes monthly.** Running the ETL more than once a month achieves nothing
except new files in `data/raw/bcrp/`. The de-duplication in dbt makes a redundant
pull harmless, not useful.

**Never re-snapshot in the middle of a sweep.** A sweep must run against exactly
one `data_version` (`training_plan.md` §4.0). Finish, then re-snapshot.

---

## 1. Environment — every session

```powershell
cd "<repo root>"
.\.venv\Scripts\Activate.ps1
```

Your prompt must show `(.venv)`. **Never `.venv-dbt`** except in §3 — dbt pins
jinja2/click/protobuf ranges that collide with MLflow's, which is exactly why
there are two environments.

Check Google credentials are alive (they expire):

```powershell
gcloud auth application-default print-access-token | Measure-Object -Character
```

A character count means fine. An error means:

```powershell
gcloud auth application-default login
```

---

## 2. ETL — pull from the BCRP API

Only when there is new data to fetch (§0).

### 2.1 Pull the raw JSON

```powershell
python -m src.data_collection.fetch_target_series
```

- One HTTP request per series, 0.3 s apart. The API accepts 10 codes per call but
  **does not return them in request order** and identifies series only by a long
  name string — one code per call makes the code→values mapping correct by
  construction. 22 requests, about 15 seconds.
- Writes `data/raw/bcrp/{pull_date}_{code}.json`. **Immutable**: a later pull adds
  files, never overwrites. These are committed to git, and the entire warehouse is
  rebuildable from them.
- Expect `22/22 series saved`. Any `✗` line names the failing code — re-run; the
  API is free and occasionally flaky.

### 2.2 Load into BigQuery

Always dry-run first. It costs nothing and tells you exactly what would change:

```powershell
python -m src.data_collection.load_to_bigquery --dry-run
```

Read the two lines it prints: how many files are already loaded, and how many rows
are new. If "new" is zero, the pull found nothing BCRP hadn't already published —
stop here, there is nothing to rebuild.

```powershell
python -m src.data_collection.load_to_bigquery
```

Append-only and idempotent: a file already loaded is skipped by `source_batch`, so
running it twice is safe. It also rewrites `raw.series_metadata` from `config.py`,
which is how a series added to the config becomes a panel column with no SQL edit.

---

## 3. dbt — rebuild the warehouse

Needed after §2, **and after any change to `config.py`** (adding, removing or
renaming a series), because the panel's column list is generated from
`raw.series_metadata` at compile time.

```powershell
deactivate
.\.venv-dbt\Scripts\Activate.ps1
cd pagos_dbt
dbt build
cd ..
deactivate
.\.venv\Scripts\Activate.ps1
```

`dbt build` runs models and tests together — a model whose test fails does not
silently feed the ones downstream.

**All tests must pass.** The one to actually read is `interior_gaps` on
`marts.series_coverage`: it asserts that no series has a missing month inside its
own coverage. Structural NULLs at the edges (Yape before 2024) are expected and
correct; a hole in the middle would silently corrupt every lag and moving average
built from that series. See `training_plan.md` §4.0.

A full build scans roughly 82 KB — seven orders of magnitude inside the free tier.

---

## 4. Snapshot — the boundary between warehouse and modelling

```powershell
python -m src.model_training.snapshot
```

Writes two files:

```
data/processed/panel_{data_version}.parquet       levels, one row per month
data/processed/series_meta_{data_version}.csv     col_name, kappa, transform, category
```

`data_version` is `MAX(pulled_at)` from the warehouse and **is the filename**, so
no run can claim a data version it did not load. After this, **nothing in
`src/model_training/` touches BigQuery again** (`training_plan.md` §4.0).

Both are gitignored — regenerable from `data/raw/bcrp/`, which is committed.

---

## 5. MLflow server — every session that trains or inspects

**Terminal one**, and leave it alone for the rest of the session:

```powershell
.\scripts\start_mlflow.ps1
```

Expect `workers : 1` in the banner and exactly **one** `Started server process`.
Then confirm before doing anything else:

```powershell
curl.exe http://127.0.0.1:5000/health     # -> OK
```

UI at `http://127.0.0.1:5000`.

> **If every request returns HTTP 500** with `AttributeError: module 'anyio' has no
> attribute 'from_thread'` — the `anyio<4.15` pin was lost. `pip install "anyio<4.15"`.
> See `mlflow_setup.md` §0.

**Open a second terminal for everything below** (§1 again: `cd`, activate).

---

## 6. Tests — before any sweep, after any change to the feature code

```powershell
pytest -q
```

Six tests guard the things that fail *silently* (`training_plan.md` §9.0): calendar
indexing at h=3, κ enforcement, strict feature-set nesting, the purge gap,
the MASE identity, and the shuffled-target leak canary.

**A red test blocks the sweep.** A leak found at step 11 invalidates everything
above it; these run in seconds.

---

## 7. Train

Dry-run first — prints run names and counts, fits nothing:

```powershell
python -m src.model_training.sweep --config configs/sweeps/s1_baselines.yaml --dry-run
```

Then for real:

```powershell
python -m src.model_training.sweep --config configs/sweeps/s1_baselines.yaml --jobs 4
```

`--jobs N` fits N configurations in parallel; **all MLflow writes stay in the parent process** (SQLite, one
writer). Omit it for a sequential run.

Sweeps, in the order the protocol requires:

| Sweep | Contents | Time | Gate before the next one |
|---|---|---|---|
| `s1_baselines` | 20 parents, five naive variants | ~4 min (GCS artifact uploads dominate) | The floor is in MLflow. **The hurdle is `naive_drift`** (training plan v1.5), not `naive_calendar` |
| `s2_proxy_grid` | 66 parents, Stage A + B | est. 30–45 min at `--jobs 6` (measured: XGBoost ≈ 4.5 min, RF ≈ 5 min, SARIMAX FS2 ≈ 7 min per config) | Check the §5 pre-registered prediction against what happened. **Decide O-10 first** |
| *holdout* | Stage C, selected config | seconds | **Once. Ever.** |
| `s3_w2021` | 12 parents, sensitivity | ~20 min | |
| `s4_wallet` | 18 parents, wallet + transfer test | ~10 min | |

Interrupting a sweep is safe: it skips runs that already finished with the same
tag set, so restarting resumes rather than duplicating.

A single configuration, without the sweep wrapper:

```powershell
python -m src.model_training.train --model xgboost --target t2 --horizon 1 --window w2019 --feature-set FS3_activity --stage cv
```

---

## 8. Read the results

The ranking table, with the hurdle column (`skill_drift`), filtered to the current protocol:

```powershell
python -m src.model_training.report --target t2 --horizon 1 --window w2019
```

`skill_drift <= 0` means persistence-plus-trend was not beaten. `--protocol 1.4` ranks an earlier protocol.

The UI to browse. For anything comparative, a notebook:

```python
import mlflow
runs = mlflow.search_runs(
    experiment_names=["26.1__t2_h1"],
    filter_string="tags.stage = 'cv' and tags.feature_set = 'FS3_activity'",
)
```

**Do not open the UI's Compare view on thousands of runs** — it loads them all.
The write-up needs reproducible tables anyway, so the notebook is the honest path.

---

## 9. Back up — before and after every long sweep

```powershell
Copy-Item mlflow/mlflow.db "mlflow/mlflow_$(Get-Date -Format yyyyMMdd_HHmm).db"
```

`mlflow.db` is the **only** copy of every run, metric and parameter. It is
gitignored, so git will not save you; only artifacts go to Cloud Storage. Thirty
seconds of insurance against three hours of runs.

---

## 10. Shut down

Ctrl-C in terminal one. Nothing else to stop — no services, no containers, nothing
running in the cloud, nothing accruing cost.

---

## Quick reference

| Task | Environment | Command |
|---|---|---|
| Activate | — | `.\.venv\Scripts\Activate.ps1` |
| Pull BCRP | `.venv` | `python -m src.data_collection.fetch_target_series` |
| Load warehouse | `.venv` | `python -m src.data_collection.load_to_bigquery --dry-run` then without |
| Rebuild models | `.venv-dbt` | `cd pagos_dbt; dbt build` |
| Snapshot | `.venv` | `python -m src.model_training.snapshot` |
| Server | `.venv` | `.\scripts\start_mlflow.ps1` |
| Tests | `.venv` | `pytest -q` |
| Sweep | `.venv` | `python -m src.model_training.sweep --config configs/sweeps/<name>.yaml` |
| Back up | — | `Copy-Item mlflow/mlflow.db "mlflow/mlflow_$(Get-Date -Format yyyyMMdd_HHmm).db"` |

## Things that have actually gone wrong

| Symptom | Cause | Fix |
|---|---|---|
| Every HTTP request returns 500, `anyio has no attribute 'from_thread'` | `anyio` upgraded past 4.15 | `pip install "anyio<4.15"` |
| `.ps1` fails with a parse error on a line that looks fine | A non-ASCII character inside a quoted string; PowerShell 5.1 reads `.ps1` as CP1252 | Keep scripts pure ASCII. Check: `Select-String -Path scripts\*.ps1 -Pattern "[^\x00-\x7F]"` |
| `pip install -r requirements.lock.txt` fails on encoding | Lock written with `>`, which emits UTF-16 | `pip freeze \| Out-File -Encoding ascii requirements.lock.txt` |
| `Could not connect to server` | Nothing is listening — the server died or never started | Check terminal one. Distinct from a 500, which means it IS listening and failing. |
| Server won't start, database locked | Orphaned process from a previous session | `Get-Process mlflow*, python \| Stop-Process` |
| dbt column missing after adding a series | Loader not re-run, so `raw.series_metadata` is stale | §2.2 then §3 |
| ETL: `UnicodeEncodeError: 'charmap' codec can't encode '✗'` | Windows console is CP1252 and the failure marker is not | `$env:PYTHONUTF8=1` (set in `.env.example`). This only masks the real failure — read the line above it |
| ETL: `JSONDecodeError: Expecting value: line 1 column 1` on **every** series | BCRP's Imperva bot protection answers with an HTML/JavaScript challenge (HTTP 200, `text/html`) instead of JSON. First seen 2026-10-04. `curl -i` on any series URL shows `Content-Type: text/html` | Retry later. Otherwise download the series by hand from the BCRP site in a browser into `data/raw/bcrp/`. **Do not** spoof headers or script a browser to defeat the challenge (training plan O-13) |

---

## 11. Git — commit, branch, publish

Branch `main`. CI (`.github/workflows/ci.yml`) runs the offline tests on every push
and pull request; it needs no secrets and must never be given any.

```powershell
git status
git add <specific paths>                  # not `git add .` -- check data/ and mlflow/ stay out
git commit -m "area: what changed and why"
git push
```

What is deliberately **not** in the repository: `mlflow/` (the experiment history lives
only on the development machine — back it up, §9), `data/processed/` (rebuildable), `.env`, `.venv*`,
and any service-account key (none exists).

Commit data pulls on their own, separate from code: `data: BCRP pull YYYY-MM-DD`.
