# Runbook — this project "ML - Pagos por Teléfono"

**Machine:** the development machine (Windows 11, PowerShell)
**Project root:** `<repo root>`
**Companions:** `docs/mlflow_setup.md` (one-time setup) · `docs/training_plan.md` (what to run and why) · `docs/project_outline.md` (architecture)

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
the MASE identity, and the shuffled-target leak canary. Since 2026-10-05 the suite also pins
the wallet window (O-12: the MASE scale, the missing seasonal reference), the seasonal-transfer
feature (§5.2: factors cannot see anything from 2024-01 on) and the `s4_wallet` spec. A few of
these log to a **throwaway SQLite store in a temp directory** — never to the server in `.env`.

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
| `s1_baselines` | 20 parents, five naive variants. **Re-run under protocol 1.6** on a refreshed snapshot (§12); 1.5 runs are ignored by the resume key and by `report.py` | ~4 min (GCS artifact uploads dominate) | The floor is in MLflow. **The hurdle is `naive_drift`** (training plan v1.5), not `naive_calendar` |
| `s2_proxy_grid` | 66 parents, Stage A + B | est. 30–45 min at `--jobs 6` (measured: XGBoost ≈ 4.5 min, RF ≈ 5 min, SARIMAX FS2 ≈ 7 min per config) | Check the §5 pre-registered prediction against what happened. **Decide O-10 first** |
| *holdout* | Stage C, selected config (`s6_holdout`: `ens3` vs `naive_drift`, §13.5) | ~1 min | **Once. Ever.** `python scripts/run_holdout.py --confirm` |
| `s3_w2021` | 12 parents, sensitivity | ~20 min | |
| `s4_wallet` | **30 parents** (plan §7.2 reconciles the old "18"): five naive variants + Ridge + XGBoost on `FS0/1/2_short` and the transfer sets `FS1s/FS2s_short`, reduced grids | ~10 min | Does **not** depend on `s2` or on a winner. Read `evaluation_status` before quoting anything |

Interrupting a sweep is safe: it skips runs that already finished with the same
tag set, so restarting resumes rather than duplicating.

A single configuration, without the sweep wrapper:

```powershell
python -m src.model_training.train --model xgboost --target t2 --horizon 1 --window w2019 --feature-set FS3_activity --stage cv
python -m src.model_training.train --model xgboost --target t4 --horizon 1 --window w2024 --feature-set FS2s_short --stage cv
```

The CLI needs nothing from `configs/models/*.yaml` (nothing reads them; `--model` goes straight to
`registry.MODELS`). On `w2024` it uses the reduced grids in `registry.W2024_GRIDS` (Ridge, XGBoost only;
any other family raises there).

---

## 8. Read the results

The ranking table, with the hurdle column (`skill_drift`), filtered to the current protocol:

```powershell
python -m src.model_training.report --target t2 --horizon 1 --window w2019
```

`skill_drift <= 0` means persistence-plus-trend was not beaten. `--protocol 1.4` ranks an earlier protocol.

The table has an **`evaluation_status`** column: `demonstration` means fewer than 8 outer folds, i.e. the run shows
the pipeline works and **ranks nothing** (plan §6.4 rule 4). For the wallet targets:

```powershell
python -m src.model_training.report --target t4 --horizon 1 --window w2024
python -m src.model_training.report --target t5 --horizon 3 --window w2024
```

`w2024` MASE is scaled by the **random walk**, not the seasonal naive (plan §6.3, O-12; param `mase_scale_period` = 1),
so it is only comparable with other `w2024` runs. Check `n_folds_nan_metric = 0` on a parent before trusting its mean.
The seasonal-transfer comparison is **same model, same target, `FS1_short` vs `FS1s_short` and `FS2_short` vs
`FS2s_short`**, read as pre-registered in plan §5.2 (Ridge is expected to show ~nothing; XGBoost at both horizons is the test).

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
| Forecast / monitor | `.venv` | `python scripts/monthly_refresh.py --dry-run --skip-etl` (§13) |
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
| ETL: `BcrpNonJsonResponse` / `BCRP returned a non-JSON body` (before 2026-10-06: `JSONDecodeError: Expecting value: line 1 column 1`) on **every** series | BCRP's Imperva bot protection answers with an HTML/JavaScript challenge (HTTP 200, `text/html`) instead of JSON. First seen 2026-10-04. `curl -i` on any series URL shows `Content-Type: text/html`. The client now retries 3 times (2, 4, 8 s), then names the status, content type and the first 200 characters of the body; the fetch stops after two such series in a row. In the monthly refresh it surfaces as `BcrpBlockedError` | Retry later. Otherwise download the series by hand from the BCRP site in a browser into `data/raw/bcrp/`. **Do not** spoof headers or script a browser to defeat the challenge (training plan O-13) |

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

---

## 12. Protocol 1.6 re-run — `s1`, `s2`, `s4`, in this order

Everything below needs a **snapshot taken after the 2026-10-05 pull** (`data/raw/bcrp/2026-10-05_*`, already committed).
The old `panel_20260905…` snapshot still works but is one month short and pre-dates the pull the plan's numbers assume.

```powershell
# environment: §1 (.venv).  Warehouse + snapshot: 
python -m src.data_collection.load_to_bigquery --dry-run     # expect new rows for 2026-07; then without --dry-run
deactivate; .\.venv-dbt\Scripts\Activate.ps1; cd pagos_dbt; dbt build; cd ..; deactivate; .\.venv\Scripts\Activate.ps1
python -m src.model_training.snapshot                        # new data_version in the filename

.\scripts\start_mlflow.ps1                                   # terminal one; second terminal for everything below
pytest -q                                                    # all green, or stop
Copy-Item mlflow/mlflow.db "mlflow/mlflow_$(Get-Date -Format yyyyMMdd_HHmm).db"

# Dry-runs first. The counts are part of the protocol: 20, 66, 30. The `snapshot :` line must name the NEW file.
python -m src.model_training.sweep --config configs/sweeps/s1_baselines.yaml --dry-run
python -m src.model_training.sweep --config configs/sweeps/s2_proxy_grid.yaml --dry-run
python -m src.model_training.sweep --config configs/sweeps/s4_wallet.yaml --dry-run

python -m src.model_training.sweep --config configs/sweeps/s1_baselines.yaml --jobs 4   # the floor, protocol 1.6
python -m src.model_training.sweep --config configs/sweeps/s2_proxy_grid.yaml --jobs 6  # hours; overnight
python -m src.model_training.sweep --config configs/sweeps/s4_wallet.yaml --jobs 4      # independent of s2
```

Notes.

- `s4_wallet` could run before `s2`: it uses no winner and no `s2` output (plan §5.2). `s1` first is still the convention, because it
  is the quickest end-to-end check that the new snapshot and the 1.6 windows behave.
- **Not in this runbook yet, on purpose:** winner selection, the Stage C holdout and `s3_w2021`. The holdout is evaluated once, ever.
- After the snapshot exists, fill the TODO fold-count tables in plan §2 / §6.2 from `Frame` (`len(X)`, `target_start`, `target_end`), not by hand.
- If a wallet config **fails** in `s4` rather than finishing, read the `FAILED:` line before re-running: `fit_config` refuses (never skips)
  a test origin without a seasonal reference, and refuses `seas_transfer` on any window that would leak it.

---

## 13. Monthly refresh and holdout

Two things live here: the **monthly refresh** (new BCRP month -> forecast with an error band -> drift check, run by GitHub
Actions or locally) and the **one-time holdout**. Neither needs anything in §1-§12 to change.

What exists (all offline-testable, none of it needs the MLflow server except the holdout):

| What | Command | Notes |
|---|---|---|
| Freeze the model | `python scripts/freeze_production.py` | Reads the protocol-1.7 `ens3` run from `mlflow/mlflow.db` read-only; writes `configs/production/t3_ens3.yaml` (hyperparameters + CV error band). Already done on snapshot `20261005T000000Z`; re-run only to reproduce |
| Forecast | `python -m src.forecasting.predict --snapshot data/processed/panel_<version>.parquet` | Fits the frozen members on all labelled rows, forecasts the latest admissible origin. `--no-write` prints only |
| Drift monitor | `python -m src.forecasting.monitor --snapshot ...` | Writes `forecasts/monitor.json`; exit 0 ok / 0 warning / 2 alert |
| Monthly refresh | `python scripts/monthly_refresh.py [--dry-run] [--skip-etl] [--from-stage X]` | fetch, load, dbt, snapshot, predict, monitor. Exit 0 done, 2 done + drift alert, 1 a stage failed. Writes `forecasts/last_refresh.json` |
| Holdout | `python scripts/run_holdout.py [--confirm]` | §13.5. **Once, ever** |

### 13.1 One-time Google Cloud setup (Workload Identity Federation, no JSON key)

GitHub Actions proves who it is with a short-lived OIDC token; Google trusts it only for repository
`rodrigogarcia92/ml-pagos-por-telefono` and lets it impersonate one service account with the minimum roles. Run once, in PowerShell,
after `gcloud auth login`:

```powershell
$PROJECT_ID = "pagos-telefono-26"
$REPO = "rodrigogarcia92/ml-pagos-por-telefono"          # owner/name, case-sensitive
$SA = "forecast-refresh@pagos-telefono-26.iam.gserviceaccount.com"

# The number (not the id) is what the provider path uses. Look it up:
$PROJECT_NUMBER = gcloud projects describe $PROJECT_ID --format="value(projectNumber)"
$PROJECT_NUMBER                                          # e.g. 123456789012 -- the <PROJECT_NUMBER> below

gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project $PROJECT_ID

# 1. Workload Identity pool + GitHub OIDC provider, restricted to the one repository
gcloud iam workload-identity-pools create github-pool `
  --project=$PROJECT_ID --location=global --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github-provider `
  --project=$PROJECT_ID --location=global --workload-identity-pool=github-pool `
  --display-name="GitHub OIDC" `
  --issuer-uri="https://token.actions.githubusercontent.com" `
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" `
  --attribute-condition="assertion.repository == '$REPO'"

# 2. The service account the workflow impersonates
gcloud iam service-accounts create forecast-refresh `
  --project=$PROJECT_ID --display-name="Monthly forecast refresh"

# 3. Minimum roles: BigQuery Data Editor on the three datasets only (not the project),
#    BigQuery Job User on the project (needed to run any query or load job)
foreach ($ds in "raw", "staging", "marts") {
  bq add-iam-policy-binding --member="serviceAccount:$SA" --role="roles/bigquery.dataEditor" "${PROJECT_ID}:$ds"
}
gcloud projects add-iam-policy-binding $PROJECT_ID `
  --member="serviceAccount:$SA" --role="roles/bigquery.jobUser"

# 4. Let identities from THAT repository (and no other) act as the service account
gcloud iam service-accounts add-iam-policy-binding $SA --project=$PROJECT_ID `
  --role="roles/iam.workloadIdentityUser" `
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/$REPO"

# 5. The provider's full resource name -- this is the value of the GCP_WIF_PROVIDER secret
gcloud iam workload-identity-pools providers describe github-provider `
  --project=$PROJECT_ID --location=global --workload-identity-pool=github-pool --format="value(name)"
# -> projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github-pool/providers/github-provider
```

Check it (nothing here prints a secret):

```powershell
gcloud iam service-accounts get-iam-policy $SA --project=$PROJECT_ID
bq get-iam-policy "${PROJECT_ID}:marts"
```

> **Scope, stated plainly.** `roles/bigquery.dataEditor` on `raw`, `staging` and `marts` can create, change and delete tables in those
> datasets, which is what the loader and `dbt build` do. It cannot touch other datasets, other projects, the MLflow bucket, or IAM.
> The service account has **no key**; deleting the provider (or the `workloadIdentityUser` binding) revokes access at once.
> If `google-cloud-bigquery-storage` is ever added to `requirements.txt`, `.to_dataframe()` will also need `roles/bigquery.readSessionUser`.

### 13.2 GitHub: secrets and one repository setting

Repository -> Settings -> Secrets and variables -> Actions -> **New repository secret** (or the CLI, from the repo folder):

| Secret | Value |
|---|---|
| `GCP_WIF_PROVIDER` | `projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github-pool/providers/github-provider` (step 5 above) |
| `GCP_SERVICE_ACCOUNT` | `forecast-refresh@pagos-telefono-26.iam.gserviceaccount.com` |

```powershell
gh secret set GCP_WIF_PROVIDER --body "projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
gh secret set GCP_SERVICE_ACCOUNT --body $SA
```

Neither is a credential by itself (they are names; the trust is the repository condition above), but there is nothing to gain from printing them.

The workflow opens a pull request with the default token, which GitHub forbids until you allow it: Settings -> Actions -> General ->
Workflow permissions -> **Read and write permissions** and **Allow GitHub Actions to create and approve pull requests**. Or:

```powershell
gh api -X PUT repos/$REPO/actions/permissions/workflow -f default_workflow_permissions=write -F can_approve_pull_request_reviews=true
```

### 13.3 Run the workflow, then switch on the schedule

Manually, the first time (Actions tab -> **Monthly refresh** -> *Run workflow*, or):

```powershell
gh workflow run monthly-refresh.yml                  # the whole chain
gh workflow run monthly-refresh.yml -f skip_etl=true # no fetch/load/dbt: new snapshot from the warehouse as it is, then forecast
gh run watch
```

What you should see, by outcome: a new BCRP month -> a pull request **`forecast: <target month>`** (raw JSON, `forecasts/`, `monitor.json`; the
body has the forecast, the 80%/90% band and the monitor status); no new month -> no PR, green job; monitor `alert` -> an issue labelled
`drift-alert`; any stage fails -> an issue labelled `refresh-failed` with the stage, the command and the last 20 lines, and a red job.
Merge the PR to publish the month.

If the first run fails at `fetch` with `BcrpBlockedError`, BCRP is serving a bot-protection page to GitHub's addresses (plan O-13). Do not
try to get around it: use §13.4.

**Enable the schedule only after one successful manual run.** In `.github/workflows/monthly-refresh.yml`, uncomment the three lines under
`# schedule:` (10th, 20th and 28th at 13:50 UTC = 08:50 Lima), commit and push to `main`. Scheduled workflows run only from the default branch,
three attempts a month are idempotent (a run that finds no new published month opens nothing), and GitHub pauses scheduled workflows after 60 days
without repository activity. Until the first successful real run, the README keeps saying the refresh is "rolling out".

### 13.4 Fallback: run it on this machine (Windows Task Scheduler)

For when BCRP blocks GitHub's IPs, or to keep the whole thing local. The task runs the same script with the project's `.venv`; `dbt` is found in
`.venv-dbt` automatically (or set `DBT_BIN`). Prerequisites: `gcloud auth application-default login` still valid (§1) and `.env` present.
From the project root:

```powershell
$root = (Get-Location).Path
$py = Join-Path $root ".venv\Scripts\python.exe"
$inner = "Set-Location -LiteralPath '$root'; & '$py' scripts\monthly_refresh.py *>> forecasts\refresh_task.log"
$tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -Command `"$inner`""

schtasks /Create /TN "pagos-telefono-monthly-refresh" /SC MONTHLY /D 10,20,28 /ST 09:00 /F /TR $tr
schtasks /Query  /TN "pagos-telefono-monthly-refresh" /V /FO LIST      # check it
schtasks /Run    /TN "pagos-telefono-monthly-refresh"                   # try it now
schtasks /Delete /TN "pagos-telefono-monthly-refresh" /F                # remove it
```

The machine must be on, and the user logged in, at that time. The task does not commit anything; after a run that made a forecast (look at
`forecasts/last_refresh.json`: `"outcome": "forecast"`), publish it yourself:

```powershell
git checkout -b forecast/<target-month>
git add data/raw/bcrp forecasts
git commit -m "forecast: <target-month>"
git push -u origin forecast/<target-month>
gh pr create --title "forecast: <target-month>" --fill
```

A single local run, by hand: `python scripts/monthly_refresh.py --dry-run` (the plan), then `python scripts/monthly_refresh.py`.
`--skip-etl` forecasts from the newest snapshot already in `data/processed/`; `--from-stage dbt` resumes after a failure.
Exit code 2 means "done, but the monitor says alert".

### 13.5 The holdout (once, ever)

`docs/training_plan.md` §11 holds the freeze entry **`PRODUCTION-FREEZE t3_ens3 v1`** with the rule written *before* the holdout is run: ens3 FS3 against
`naive_drift` on the final 12 target months (2025-08 ... 2026-07 on snapshot `20261005T000000Z`); MASE, MAPE and the paired per-month difference
+/- SE; published whatever it is; nothing re-tuned, re-selected or re-run afterwards. The command refuses unless that marker is in the plan, you
pass `--confirm`, and MLflow has no finished holdout run for the two parents on this `data_version`.

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\start_mlflow.ps1                      # terminal one (§5); then a second terminal for the rest
Copy-Item mlflow/mlflow.db "mlflow/mlflow_$(Get-Date -Format yyyyMMdd_HHmm).db"      # §9, before
pytest -q                                       # green, or stop
python -m src.model_training.sweep --config configs/sweeps/s6_holdout.yaml --dry-run # must show 2 runs
python scripts/run_holdout.py                   # preview: plan + the three guards, exit 0
python scripts/run_holdout.py --confirm         # THE run: ~1 minute, 2 parents
```

It prints the comparison table and two blocks of text: one row for §11 and one paragraph for the README. Paste both **as printed**, commit, and
stop. If ens3 does not beat `naive_drift`, the README says so and `naive_drift` becomes the production fallback; that is the rule, not a judgement call.
Back up `mlflow.db` again afterwards (§9). A holdout run that *fails* is not recorded, so the guards allow a re-run; one that *finishes* is final.

The production forecast refits the frozen members on **all** labelled rows, including the holdout months. That is a production fit (it scores
nothing), and it is also why the holdout can be evaluated only once and before anyone treats the 2025-08 ... 2026-07 errors as a result.
