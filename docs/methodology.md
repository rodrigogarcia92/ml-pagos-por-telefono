# Methodology

How the forecast is built, evaluated and operated, and which decisions were fixed in advance. Section numbers in code comments (e.g. "plan §7.4") refer to the internal training plan; the relevant rules are summarised here.

## 1. Question and proxy

**Question.** How many phone-based transfers will Peruvians make in the coming months, and is that volume predictable enough to plan around?

**Data.** Public monthly series from the central bank (BCRP) through its API: payment-system counts and amounts, plus macro variables (cash in circulation, monthly GDP index, CPI, exchange rate, policy rate, formal income, dollarisation). The series are loaded to BigQuery and modelled with dbt (staging → marts); training reads a frozen parquet snapshot of the panel, never the warehouse, so every run names the exact `data_version` it used.

**Proxy.** Yape and Plin publish no volumes of their own, and the wallet series only start in 2024-01. The primary target is therefore the *aggregate intrabank transfer count* (`n_transf_intra_agg`), which has history back to 2013 and contains the wallets' flows. The wallet series (Yape + Plin) are modelled separately on a short window and are labelled a demonstration where there are too few folds to rank anything.

**Targets.** The target is the cumulative log difference to the last published level, `z_h = log n(target) − log n(anchor)`. Models are fitted on `z`; every metric is computed on the reconstructed level `n̂ = n(anchor)·exp(ẑ)` against the actual level.

## 2. Publication lags and the target-month rule

Each series `j` has a publication lag `κ_j` (months between the end of a month and its first release). The information available when a forecast is made at the close of month `t` is `{x_s : s ≤ t − κ_j}`.

| Series | κ |
|---|---|
| Payments family (target and its own lags) | 2 |
| `pbi_idx` (monthly GDP), `ingreso_formal` | 2 |
| `circulante`, `ipc` | 1 |
| `tipo_cambio`, `tasa_referencia` | 0 |
| Google Trends (tested, rejected) | 0 |

The payments lag was first assumed to be 1 month. Two pulls a week apart both ended one month earlier than expected, so it was corrected to 2 and the whole timing convention was restated.

With `κ = 2` the anchor is `n(t−2)` and the **target month is `t + h − 2`**. Stochastic features (own lags, macro) are indexed at `t − k` with `k ≥ κ_j`; deterministic features (calendar, holidays) are indexed at the *target* month, because they are known in advance. The dataset builder derives the target month from `κ` rather than hard-coding it, and **refuses** to build a feature that violates its series' `κ` — a missing or too-small `κ` would otherwise yield excellent backtests and a worthless model. `κ` is declared once, in the series registry, and fails at import if absent or negative.

## 3. Data window and the locked holdout

| Window | First target month | Use |
|---|---|---|
| `w2019` | 2019-01 | Primary: wallets at critical mass, includes the COVID break and the Plin launch |
| `w2021` | 2021-01 | Sensitivity: post-pandemic regime only |
| `w2024` | 2024-01 | Wallet targets (real Yape/Plin data); uses the shorter `FS*_short` feature sets |

A window is bounded by its first *target* month and has no end bound; the origin range is derived from `κ` and `h`.

The **final 12 target months** (6 on `w2024`) are a locked holdout. All selection, tuning and feature-set comparison happens on the earlier rows (the CV period). The holdout is evaluated **once**, for the selected configuration only, under a rule written down before it was run (§9).

## 4. Cross-validation and leakage guards

- **Expanding window**, never a random split: with autocorrelated data a random split is a guaranteed fake result.
- **Minimum training size** per window: 36 (`w2019`), 24 (`w2021`), 12 (`w2024`).
- **Purge gap** of `h − 1` rows between the end of training and each test origin.
- **Scalers and imputers are fitted inside each fold**, on training rows only.
- **Tuning** uses an inner expanding-window CV inside the CV period (the last 10 origins); hyperparameters are then frozen for the outer folds. The gap between CV and holdout performance is itself reported, as a measure of selection optimism.
- **Tests that guard silent failures:** calendar features indexed at the target month; the `κ` guard; a shuffled-target canary that must score no better than chance (end-to-end leak detector); strict nesting of the feature sets; a scale-invariance test for the Trends features.
- **Reporting rule:** any result with fewer than 8 outer folds is tagged `demonstration`, not `evaluation`. Fold counts are an upper bound on evidence, since expanding folds share training data and overlapping targets at `h ≥ 3` reduce independent information further.

## 5. Metrics

All metrics are computed on reconstructed levels.

- **MASE (primary).** MAE divided by the in-sample MAE of the seasonal naive (period 12), computed on each fold's training rows. On `w2024`, where a fold can have only 12 training levels, the scale is the random walk's in-sample MAE (period 1), chosen from the window alone and logged. MASE values from different windows are never compared.
- **MAPE** is reported, not optimised.
- **MAE, RMSE** (pooled over all test points) give business-legible magnitudes.
- **Paired differences ± SE.** Models are compared fold by fold on the same test origins; the difference in per-fold MASE has standard error `sd / √n_folds`. "Tied" means the difference is within 1 SE. The same paired comparison on the holdout is made per month.
- Uncertainty on model comparisons is the paired SE above. Folds share training data and neighbouring months are serially correlated, so the SE is a rough guide, not an exact interval.
- **Ranking stability check (supporting, not decisive).** To see how stable the ranking is under resampled history, the 41 paired per-fold MASE values (target `t3`, `w2019`, snapshot `20261005T000000Z`) of the nine candidates in the selection chart were resampled with a moving block bootstrap: overlapping blocks of 6 consecutive folds, 5,000 replicates, fixed seed, the same folds drawn for every model so comparisons stay paired. The ensemble's mean MASE was below naive drift's in 99.6% of replicates; the 90% percentile interval of the mean paired difference (ensemble − naive drift) is [−0.243, −0.058]; the ensemble had the lowest mean MASE most often (36%, ahead of XGBoost on all macro variables 21% and on macro plus policy stance 20%), and naive drift never did. The intervals depend slightly on the seed (the headline share moves by about ±0.1 points). This check supports, but did not drive, the pre-registered decision rule, which remains the paired SE above. Reproduce with `scripts/selection_bootstrap.py` (read-only on MLflow).

## 6. The hurdle

The baseline to beat is **naive drift** (last level plus the average growth so far). Five naive variants were run first; on a series that grew roughly forty-fold, a seasonal reference gives up about a year of growth and is a poor hurdle. Drift ranked first in all four horizon × window cells of the baseline sweep, so it replaced the originally named calendar-naive as the hurdle. `skill_drift = 1 − MAE_model / MAE_drift` is reported for every run; `skill_drift ≤ 0` means the model has not beaten persistence plus trend. It is a reporting column and changes no fit, fold or ranking.

## 7. Feature sets and model families

Feature sets are strictly nested and declared as explicit column lists in a config file; the nesting is asserted by a test.

| Set | Adds | Columns (tree) |
|---|---|---|
| `FS0_calendar` | calendar and COVID controls | 6 |
| `FS1_autoregressive` | own history (differences and moving averages at admissible lags) | 11 |
| `FS2_cash` | cash in circulation | 17 |
| **`FS3_activity`** | monthly GDP index | **22** |
| `FS4_prices` | CPI | 28 |
| `FS5_macro_full` | exchange rate, formal income, policy rate, dollarisation | 53 |
| `FS3_gt` | `FS3` + Google Trends index (never swept; see §8) | 25 |

`FS*_short` variants drop every 12-month term so that the 30-month wallet window keeps enough rows. A seasonal-transfer test supplies seasonal factors measured on the aggregate series (pre-registered before any wallet run).

**Model families:** naive variants (last, drift, seasonal, calendar, seasonal-drift); Ridge and ElasticNet; SVR (RBF); Random Forest; XGBoost (CPU `hist`); SARIMAX with a fixed exogenous-variable rule; and **`ens3`**, the equal-weight mean of the SVR, Random Forest and XGBoost forecasts of `z`, each member tuned independently on the same inner folds. Equal weights were fixed in advance because weights estimated from ~40 folds would fit noise.

## 8. Decision rules fixed in advance

Written before the corresponding results were read. Changing a rule after the fact is a new protocol version, not an amendment (D3).

**Predictions** (checked after the sweep, before any holdout)

| # | Prediction |
|---|---|
| P1 | Trends help more at 5–6 months ahead than at 3 |
| P2 | Trends reduce the under-forecast bias on 2022 target months |
| P3 | `ens3` is within 1 SE of its best member and has lower fold-to-fold spread than each member |
| P4 | `skill_drift` falls with horizon (3 > 5 > 6 months) |

**Decisions**

| # | Rule |
|---|---|
| D1 | Trends enter the production set only if `ens3` on `FS3_gt` beats `ens3` on `FS3` by more than 1 SE at ≥ 2 of 3 horizons; otherwise reported as a negative result |
| D2 | The ensemble becomes the production family only if it is not worse than its best member by more than 1 SE at every horizon |
| D3 | No change to these rules after the sweep has been read |

**Google Trends data gate** (on the CV period only, thresholds fixed before any Trends value was examined)

| Gate | Test | Pass condition |
|---|---|---|
| G1 coverage | months from 2017-09 with index < 5 | ≤ 6 |
| G2 stability | correlation of Δlog index between two pulls on different days | ≥ 0.90 |
| G3 relevance | max cross-correlation of Δlog index with Δlog target at leads 0–3 | ≥ 0.20, positive |

**Outcome.** G1 failed (46 months below 5; the index is literally 0 until 2021-05) and G3 failed (max +0.195 at lead 2, with signs alternating across leads), so Trends are a **negative result** and D1 resolves to "not adopted". The apparent rise of Plin from 0 to 10 in 2023 is a low-volume threshold artifact in the index, not adoption. The Trends pipeline stays in the repository, unswept, and is opt-in in dbt. FS4/FS5 are likewise a pre-registered negative result: out-of-sample permutation importance of their blocks is zero within noise.

## 9. How the production model was chosen and frozen

The production feature set (`FS3_activity`) was designated from the CV sweep: removing any FS3 block worsened at least one non-linear family, and removing the cash, activity or COVID block worsened all three. Then, on CV only (snapshot `20261005T000000Z`):

- `ens3` on `FS3_activity`, `t3` (3 steps ahead, `κ = 2`), `w2019`, ranked first (MASE 0.265). It is tied within 1 SE with SVR on `FS3` (the named fallback) and with XGBoost on `FS5`; `FS5` stays excluded as a negative result. D2 is satisfied.
- The frozen artefact holds the three members' encodings and hyperparameters, the 22 columns, and the **error band**: empirical quantiles (p05/p10/p50/p90/p95) of `log(actual / forecast)` over the 41 CV folds (target months 2022-03 … 2025-07; the holdout months are asserted absent). CV MAPE is 4.39%.
- The holdout rule was recorded *before* the holdout was run: **"Stage C compares exactly two forecasts on the final 12 target months: ens3 FS3 vs naive_drift. Reported: MASE, MAPE and the paired per-month difference ± SE for both. The result is published whatever it is. Nothing is re-tuned, re-selected or re-run after it is seen; if ens3 does not beat naive_drift, the README says so and naive_drift becomes the production fallback."**
- The holdout run takes the frozen hyperparameters as they are (no tuning step) and refuses to run unless the freeze marker below is in the decision log, a confirmation flag is given, and no finished holdout run exists for these configurations.

**Holdout result (once, 12 target months 2025-08 … 2026-07).**

| | `ens3` on FS3 | naive drift |
|---|---|---|
| MASE | 0.213 | 0.568 |
| MAPE | 2.82% | 7.59% |
| Paired difference (ens3 − drift) | MASE −0.355 ± 0.109 | MAPE −4.76 ± 1.51 pp |

The frozen model beat the hurdle on the holdout. Nothing was changed afterwards. Production forecasts refit the frozen members on all labelled rows; that is a production fit and scores nothing.

## 10. Monthly refresh, error band and drift alert

A scheduled GitHub Actions workflow runs on the 10th, 20th and 28th of each month (BCRP publishes on no fixed day). It authenticates to Google Cloud with Workload Identity Federation — no stored keys — and runs the chain: BCRP pull → BigQuery → dbt → snapshot → forecast.

- If a new month has been published, it refits the frozen members, forecasts the next target month, attaches the **error band** (point forecast × `exp` of the stored quantiles, giving an 80% and a 90% interval), and opens a pull request with the new raw data and forecast.
- If no new month is available it does nothing. If a step fails it opens an issue.
- Once a forecast's actual value is published it is scored, and a **drift alert** fires when the forecast misses by more than 10% in two consecutive calendar months. The monitor also reports a rolling 12-month MAPE and the empirical coverage of the 80% and 90% bands against their nominal levels.

## 11. Decision log

Marker for the freeze entry (read by the holdout guard): **PRODUCTION-FREEZE t3_ens3 v1**.

| Date | Decision | Evidence |
|---|---|---|
| 2026-09-05 | Payments publication lag set to κ = 2 | Two pulls a week apart ended one month earlier than a 1-month lag implies; all 13 payment-family series agree |
| 2026-10-04 | **O-11 closed:** hurdle is naive drift; `skill_drift` added | Baseline sweep (20 runs): drift first in 4 of 4 cells (MASE 0.340 / 0.402 / 0.208 / 0.233), calendar naive last (≈ 1.8–2.9) |
| 2026-10-05 | **O-10 closed:** windows bound the target month; origin range derived from κ and h | Origin-bounded windows left the last published month unused and shifted the holdout by a different amount per horizon; no model had been selected |
| 2026-10-05 | **O-12 closed:** MASE scale on `w2024` is the random walk (period 1) | With 12 minimum training levels the period-12 scale was uncomputable, so every `w2024` run failed on its first fold; found by reading the code, not from a result |
| 2026-10-05 | **FS3 is the production feature set**; FS4/FS5 recorded as a negative result | Block-removal ablation and out-of-sample permutation importance on CV; FS5 weak-prior blocks zero within noise |
| 2026-10-05 | `ens3`, P1–P4 and D1–D3 pre-registered | Chosen after exploratory runs on CV folds (ensemble −0.009 ± 0.019 MASE vs SVR at h=3); alternatives tried and rejected are disclosed |
| 2026-10-06 | **O-14 closed:** Google Trends rejected (G1 and G3 fail); D1 = not adopted | Gate report on the 2026-10-05 pull, CV period only; thresholds fixed before the data was examined |
| 2026-10-06 | **`ens3` frozen** as `PRODUCTION-FREEZE t3_ens3 v1` with the holdout rule written first | CV MASE 0.265, first of the field and tied with SVR on FS3; error band from 41 CV folds |
| 2026-10-06 | **Holdout passed** | 12 target months: MASE 0.213 vs 0.568, MAPE 2.82% vs 7.59%; paired MASE difference −0.355 ± 0.109 |
| 2026-10-06 | Reporting fix: true pooled RMSE | Per-fold RMSE with one test point equals MAE; the pooled value is computed from fold errors, no run repeated |
| 2026-10-07 | **Go-live:** monthly refresh enabled; first automated forecast (target month 2026-10) published | Full-chain workflow run green; schedule on the 10th, 20th and 28th |
