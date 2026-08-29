# Correlation findings — Yape/Plin vs aggregate payment series

**Roadmap step 5**
**Date:** 2026-08-24
**Source:** `marts.monthly_panel` (BigQuery), 25 series
**Notebook:** `notebooks/02_correlation.py`

---

## 1. Summary

Two aggregate BCRP payment series were tested as candidate features for forecasting Yape + Plin volumes. Both looked strongly related in levels. **Only one survived de-trending.**

| Relationship (vs `n_yape_plin`) | Levels *r* | Growth *r* | Reading |
|---|---|---|---|
| `n_transf_intra_agg` | 1.00 | **0.98** | Near-identity — not an independent feature |
| `v_transf_intra_agg` | 0.70 | **0.63** | Real, moderate, and the most interesting |
| `n_dinero_electronico` | 0.93 | **0.11** | Spurious — shared trend only |
| `v_dinero_electronico` | 0.94 | **0.21** | Spurious — shared trend only |

**Headline:** electronic money (*dinero electrónico*) and alias-based wallet transfers (Yape/Plin) are **not** substitutes and do not co-move month to month, despite both rising over 2013–2026. Aggregate intrabank transfers, by contrast, are effectively the same variable as the target.

---

## 2. Method

Every series in this set trends upward. Two trending series correlate near 1.0 whether or not they are related — the correlation measures the trend, not the relationship. Correlations were therefore computed twice:

1. **Levels** — reported only to demonstrate how misleading they are.
2. **Month-over-month log growth**, `log(x_t) − log(x_{t−1})` — differencing removes the shared trend, leaving co-movement in the deviations. This is the measure acted on.

**Window:** the overlap between the wallet split (from Jan 2024) and the aggregates (from Jan 2013) gives 30 monthly observations, 29 after differencing.

**Precision:** at *n* = 29 the 95% interval around any *r* is roughly ±0.35 (Fisher *z*). Concretely:

- *r* = 0.11 → CI ≈ [−0.27, 0.46] — indistinguishable from zero
- *r* = 0.63 → CI ≈ [0.34, 0.81] — real, imprecise
- *r* = 0.98 → unambiguous

Nothing in the 0.2–0.5 band should be over-read.

---

## 3. Findings

### 3.1 `n_transf_intra_agg` — 0.98 is a warning, not a green light

A 0.98 correlation on *growth rates* is not co-movement; it is near-identity. Yape and Plin do not merely correlate with intrabank transfers — they substantially **are** intrabank transfers. In Jun 2026 the aggregate stood at 1,114.2 million operations, and the wallet series track it almost month for month.

**Consequence — exclude from Phase 1 features.** Two independent reasons:

1. *Circularity.* The target is a component of the regressor.
2. *Availability.* BCRP publishes the aggregate and the wallet split in the same release, so at forecast time next month's aggregate is no more observable than next month's target. A feature that cannot be observed before the target moves is not a feature.

**Consequence — promote to Phase 2 panel backbone.** This series provides **163 months** of the exact phenomenon being forecast, against 30 months at wallet level. That is what makes the planned transfer-learning approach defensible: learn the shape of Peru's digital-transfer adoption curve from the long aggregate, then specialise to the Yape/Plin decomposition.

### 3.2 `dinero_electronico` — a textbook spurious correlation

0.93 in levels; **0.11 in growth**. No detectable month-to-month relationship.

The timing explains it. Both series rise over 2013–2026, which is all the levels correlation captured — but they rise at different times:

- **Intrabank transfers:** flat at ~15–20M operations through 2013–2018; inflection around 2019–2020; steady compounding thereafter to 1,114M.
- **Dinero electrónico:** essentially zero until 2019; a small COVID-era bump; **flat to declining through 2021–2023 — precisely when Yape grew fastest**; takes off only from 2024, reaching 19.7M.

**Real-world reading:** prepaid e-money and alias-based bank transfers are not competing for the same behaviour. Peruvians adopting Yape were not switching away from e-money wallets — they were switching away from **cash**.

Internal consistency check: `n_dinero_electronico ↔ v_dinero_electronico` = 0.97 on growth. The series measures *something* coherently; that something is simply unconnected to the target.

**Scale context:** 19.7M e-money operations vs 1,114M intrabank transfers in Jun 2026 — about **1.8%**.

### 3.3 Value ≠ count — the most economically interesting result

For the *same* aggregate flow, the **count** correlates with wallet growth at 0.98 while the **value** correlates at only 0.63. The two dimensions of one series behave differently. `n_transf_intra_agg ↔ v_transf_intra_agg` is itself only 0.66 on growth.

**Hypothesis.** The *count* of intrabank transfers is dominated by an enormous number of very small wallet payments. The *value* is dominated by a comparatively small number of large transfers — business payments, rent, transfers between one's own accounts — driven by different factors.

If correct, **average ticket size should be falling sharply**. That would mean Peru is not merely migrating existing payments onto digital rails; it is **creating transactions that previously did not exist as transactions at all**, because they were coins changing hands.

This is a cash-substitution story, not an instrument-substitution story — and it is consistent with §3.2, where the instrument-substitution candidate showed no relationship.

**Test:**

```sql
SELECT obs_date,
       SAFE_DIVIDE(v_transf_intra_agg, n_transf_intra_agg) AS ticket_promedio
FROM `pagos-telefono-26.marts.monthly_panel`
WHERE n_transf_intra_agg IS NOT NULL
ORDER BY obs_date;
```

A monotonic decline from 2013 to 2026 confirms the hypothesis. **Not yet run.**

---

## 4. Caveats

**The 2024 structural break in `dinero_electronico`.** A series flat near zero for eleven years that then grows ~15× in thirty months warrants scrutiny before modelling. Candidate explanations: genuine adoption, regulatory change, new issuers — or **a change in what BCRP counts**. Note that the wallet-split series also begin in January 2024. Two series changing behaviour in the same month is a coincidence worth ruling out against BCRP's *Reporte del Sistema Nacional de Pagos*.

**Sample size.** 29 growth observations. The 0.98 and 0.11 are far enough from the middle to be safe; the 0.63 is directionally reliable but imprecise.

**Unverified units.** Aggregate series are explicitly *millones* (verified against the API). The target and CCE series' scales are asserted by BCRP's table headers but not independently confirmed — see §5. Any ratio mixing the two groups is currently unit-unsafe.

---

## 5. Variable dictionary

Column names follow one convention: **`n_` = número** (count of operations), **`v_` = valor/monto** (amount transacted). Everything below is a column in `marts.monthly_panel`.

### 5.1 Targets — the wallet split (`category = 'target'`)

Jan 2024 onwards, ~30 observations. BCRP's *"Pagos inmediatos con alias"* table: transfers where the payer identifies the payee by phone number or alias rather than by account number.

| Column | Code | Meaning in the real world |
|---|---|---|
| `n_transf_intra_yape` | PN42672EM | Yape payments where payer and payee bank with the **same** institution (mostly BCP↔BCP). The dominant Yape flow. |
| `n_transf_intra_plin` | PN42673EM | Same, for Plin (the Interbank/BBVA/Scotiabank consortium wallet). |
| `n_transf_inter_visa_yape` | PN42677EM | Yape payments **crossing** banks, routed via Visa Direct. Grew after interoperability was mandated. |
| `n_transf_inter_visa_plin` | PN42678EM | Same, for Plin. |
| `v_transf_intra_yape` | PN42662EM | Soles moved by the intrabank Yape flow. |
| `v_transf_intra_plin` | PN42663EM | Soles moved by the intrabank Plin flow. |
| `v_transf_inter_visa_yape` | PN42667EM | Soles moved by interbank Yape. |
| `v_transf_inter_visa_plin` | PN42668EM | Soles moved by interbank Plin. |

*Intra vs inter matters:* intrabank transfers settle inside one bank's ledger and are near-costless to the bank; interbank transfers require settlement infrastructure. The intra/inter mix is a direct read on how genuinely interoperable Peru's wallet market has become.

### 5.2 Aggregate payment instruments (`category = 'pagos_agregados'`)

Jan 2013 onwards, ~163 observations. BCRP's *"Instrumentos de pagos de alto y bajo valor"*, **Bajo valor** block. Units verified: *millones*.

| Column | Code | Meaning in the real world |
|---|---|---|
| `n_transf_intra_agg` | PN42200EM | **All** intrabank transfers system-wide, every bank, every channel — the parent flow the Yape/Plin split carves up. 14.7M (Jan 2013) → 1,114.2M (Jun 2026). |
| `v_transf_intra_agg` | PN42171EM | Soles moved by that flow. Behaves differently from the count — see §3.3. *Code inferred from the monto/número offset; confirm on next pull.* |
| `n_dinero_electronico` | PN42209EM | Operations using **prepaid e-money** — regulated stored-value balances held outside a bank account. A distinct instrument from wallet transfers. 0 (2013) → 19.7M (Jun 2026). |
| `v_dinero_electronico` | PN42180EM | Soles moved through e-money. → S/ 2,279M (Jun 2026). Implied average ticket ≈ S/ 116. |

*Not pulled, available if the channel split becomes interesting:* PN42172EM / PN42201EM (intrabank via **non-presential** channels — app and web) and PN42173EM / PN42202EM (**presential** — branch and agent). The non-presential share would be a clean digital-adoption measure.

### 5.3 CCE clearing-house series (`category = 'complementary'`)

Jan 2024 onwards only — **shorter than intended**. These were included to extend history to 2017; they do not. Retention pending a correlation check.

| Column | Code | Meaning in the real world |
|---|---|---|
| `n_cce_cheques` | PN42230EM | Cheques cleared through the CCE. A declining legacy instrument — useful as a *disappearing* baseline. |
| `n_cce_credito` | PN42232EM | Credit transfers via the clearing house (count). Traditional interbank transfers, typically next-day. |
| `v_cce_credito` | PN42231EM | Amount moved by those transfers. |
| `n_cce_inmediatas` | PN42234EM | Immediate interbank transfers via CCE (count) — the real-time rail Plin and interoperable Yape ride on. |
| `v_cce_inmediatas` | PN42233EM | Amount moved by immediate transfers. |
| `v_alias_intra_tot` | PN42661EM | Total value of intrabank alias payments before the wallet split — i.e. Yape + Plin + any other alias scheme combined. |

### 5.4 Macro context (`category = 'macro'`)

Jan 2010 onwards, ~200 observations.

| Column | Code | Meaning in the real world |
|---|---|---|
| `ipc` | PN38705PM | Consumer price index. Nominal transfer *values* rise with prices even if real activity is flat — required to deflate `v_` series. |
| `tipo_cambio` | PN01246PM | Soles per US dollar, monthly average. Peru is partly dollarised; the rate shifts which currency people transact in. |
| `tasa_referencia` | PD04722MM | BCRP policy rate (%). The monetary-conditions variable — affects credit and, indirectly, transaction volumes. |
| `pbi_idx` | PN01770AM | Monthly GDP index (2007 = 100). The best available proxy for real economic activity at monthly frequency. |
| `dolarizacion_liquidez` | PN00025MM | Share of liquidity held in foreign currency (%). Falling dolarisation means more soles transacting domestically. |
| `circulante` | PN00048MM | Currency in circulation, MN (millones S/). **The direct cash-substitution counterpart.** If wallet payments displace cash, this should decelerate relative to GDP. |
| `ingreso_formal` | PN37696PM | Average nominal income, formal private sector (S/). Household capacity to transact. |
| *empleo* | **TBD** | Unresolved — BCRP's employment series are thin; INEI's ENAHO is the likely source. See `docs/data_sources.md` §7. |

`circulante` is worth singling out: given §3.3's cash-substitution hypothesis, it is the natural test variable. If wallets are replacing coins, currency in circulation should grow more slowly than nominal GDP over the same period.

### 5.5 Derived columns (built in the notebook, not in the warehouse)

| Column | Definition |
|---|---|
| `n_yape_plin` | Sum of the four target `n_` series — total wallet operations. |
| `v_yape_plin` | Sum of the four target `v_` series — total wallet value. |

`marts.fct_wallet_metrics` holds the warehouse-side equivalents plus `yape_share_valor`, `yape_share_numero`, `ticket_promedio_raw` and month-over-month growth.

---

## 6. Decisions

| Series | Phase 1 (baseline) | Phase 2 (panel) |
|---|---|---|
| `n_transf_intra_agg` | **Exclude** — circular and unavailable at forecast time | **Core backbone** — 163 months of the target phenomenon |
| `v_transf_intra_agg` | Investigate — the count/value divergence is the live question | Include |
| `n_dinero_electronico` | **Exclude** — no growth relationship | Optional control for the cash-substitution argument |
| `v_dinero_electronico` | **Exclude** | Optional control |
| CCE series | Re-evaluate — likely redundant vs the aggregates | Probably drop |
| `circulante` | **Promote** — direct test of cash substitution | Include |

---

## 7. Open questions

1. **Run the average-ticket query (§3.3).** The single highest-value next step; it either confirms or kills the cash-substitution narrative.
2. **Verify the Jan 2024 break** in `dinero_electronico` against BCRP's payment-system report.
3. **Confirm PN42171EM** — the only code in the set inferred rather than verified.
4. **Resolve target-series units** so ratios mixing target and aggregate series become safe.
5. **Decide the CCE series' fate** once (1)–(4) are settled.
6. **Employment series** — still open, `docs/data_sources.md` §7.

---

## 8. Reproducing

```powershell
# .venv
python -m src.data_collection.fetch_target_series
python -m src.data_collection.load_to_bigquery

# .venv-dbt, from pagos_dbt/
dbt build

# .venv
python -m notebooks.02_correlation
```
