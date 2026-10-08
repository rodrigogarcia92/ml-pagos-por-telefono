"""Series registry — the single source of truth.

Mirrors the `raw.series_metadata` table defined in docs/data_sources.md —
keep the two in sync when series are added or the employment series (§7 of
that doc) is resolved.

WHAT EACH SERIES CARRIES, AND WHY IT CARRIES IT HERE
----------------------------------------------------
    code, col_name, description, category, frequency
        Identity and provenance. Flow into raw.series_metadata and from there
        into the dbt models, which generate the panel's column list from them.

    kappa       publication lag in months (plan §4.1)
    transform   "log_diff" or "simple_diff" (plan §4.2)
        Modelling assumptions. These do NOT go to BigQuery. `snapshot.py`
        copies them into data/processed/series_meta_{data_version}.csv, and
        `dataset.py` reads them from there and refuses to build any feature
        that violates kappa.

Why the split: coverage — when a series starts, ends, and whether it has holes
— is a FACT computed from the data, so it is a dbt model (marts.series_coverage).
kappa is an ASSUMPTION a human declared and can be wrong about; it belongs with
the protocol that acts on it, versioned in git, changeable without an ETL run
and a warehouse rebuild. Facts in the warehouse, assumptions in the code.
"""

# --------------------------------------------------------------------------- #
# Transform vocabulary — plan §4.2
#
#   log_diff     multiplicative series, strictly positive. Δ = Δ log x.
#   simple_diff  a percent or a bounded share. Δ = x_t − x_{t−1}, in
#                percentage points. A rate can approach zero, where a log
#                difference is undefined or unstable, and basis points are
#                the unit anyone actually reasons in.
# --------------------------------------------------------------------------- #
TRANSFORMS = {"log_diff", "simple_diff"}

# Publication lag for the wallet/payment families.
#
# CHANGED 1 -> 2 on 2026-09-05. O-9 CLOSED, with evidence.
#
# The pre-registered protocol assumed the last observed payments month was t-1.
# Two independent pulls say otherwise:
#
#     pull 2026-08-29  ->  payments last_obs 2026-06
#     pull 2026-09-05  ->  payments last_obs 2026-06     (still)
#
# Seven days apart, and on 5 September BCRP still has not published July. A
# one-month lag would have put July out in early August. Every one of the 13
# payment-family series agrees, while every macro series matches its declared
# kappa exactly -- so this is BCRP's release calendar, not a broken pull.
#
# WHAT THIS CHANGES. At the close of month t the anchor is n_{t-2}, not n_{t-1},
# and the target month becomes t - kappa + h (dataset.py). At h=1 the product is
# still a genuine nowcast -- "last month's figure before the statistics office
# publishes it" -- it is simply one month further back than the plan assumed.
#
# Logged as a deviation in plan §11.
KAPPA_PAYMENTS = 2

TARGET_SERIES = [
    {"code": "PN42672EM", "col_name": "n_transf_intra_yape",      "description": "Número — Intrabancarias — Yape",                    "category": "target", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42673EM", "col_name": "n_transf_intra_plin",      "description": "Número — Intrabancarias — Plin",                    "category": "target", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42677EM", "col_name": "n_transf_inter_visa_yape", "description": "Número — Interbancarias (Visa Direct) — Yape",      "category": "target", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42678EM", "col_name": "n_transf_inter_visa_plin", "description": "Número — Interbancarias (Visa Direct) — Plin",      "category": "target", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42662EM", "col_name": "v_transf_intra_yape",      "description": "Valor (S/) — Intrabancarias — Yape",                "category": "target", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42663EM", "col_name": "v_transf_intra_plin",      "description": "Valor (S/) — Intrabancarias — Plin",                "category": "target", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42667EM", "col_name": "v_transf_inter_visa_yape", "description": "Valor (S/) — Interbancarias — Yape",                "category": "target", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42668EM", "col_name": "v_transf_inter_visa_plin", "description": "Valor (S/) — Interbancarias — Plin",                "category": "target", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
]

# kappa confirmed 2026-08-30 (O-1 closed) and cross-checked 2026-09-05 against
# marts.series_coverage.months_behind_panel_edge, which agrees for every row:
# the policy rate is current, monthly macro is one month behind, monthly GDP
# and formal income are two.
MACRO_SERIES = [
    {"code": "PN38705PM", "col_name": "ipc",                   "description": "Inflación (IPC)",                                      "category": "macro", "frequency": "mensual", "kappa": 1, "transform": "log_diff"},
    {"code": "PN01246PM", "col_name": "tipo_cambio",           "description": "Tipo de cambio nominal promedio (S/ por US$)",         "category": "macro", "frequency": "mensual", "kappa": 0, "transform": "log_diff"},
    {"code": "PD04722MM", "col_name": "tasa_referencia",       "description": "Tasa de Referencia de la Política Monetaria",          "category": "macro", "frequency": "mensual", "kappa": 0, "transform": "simple_diff"},
    {"code": "PN01770AM", "col_name": "pbi_idx",               "description": "PBI mensual (índice 2007=100)",                        "category": "macro", "frequency": "mensual", "kappa": 2, "transform": "log_diff"},
    {"code": "PN00025MM", "col_name": "dolarizacion_liquidez", "description": "Coeficiente de Dolarización de la Liquidez (%)",       "category": "macro", "frequency": "mensual", "kappa": 1, "transform": "simple_diff"},
    {"code": "PN00048MM", "col_name": "circulante",            "description": "Circulante — Emisión Primaria MN (millones S/)",       "category": "macro", "frequency": "mensual", "kappa": 1, "transform": "log_diff"},
    {"code": "PN37696PM", "col_name": "ingreso_formal",        "description": "Ingreso promedio sector formal privado — Nominal (S/)", "category": "macro", "frequency": "mensual", "kappa": 2, "transform": "log_diff"},
    # Empleo: TBD — see docs/data_sources.md §7 (BCRP series look thin, checking INEI ENAHO next)
]

# --------------------------------------------------------------------------- #
# Complementary series — REDUCED 2026-09-05 from six to three.
#
# The coverage audit settled what these are worth. All six start 2024-01, not
# the 2010-01 they were requested from, so they never delivered the pre-2024
# history they were included for — PAGOS_AGREGADOS does that instead. None of
# them appears in any feature set FS0–FS5b.
#
# DROPPED (three): n_cce_cheques, n_cce_credito, v_cce_credito. Cheques and
# ordinary credit transfers are different rails from wallet payments, no open
# question depends on them, and their raw JSON stays committed in
# data/raw/bcrp/ so re-adding one is a single line plus a pull.
#
# KEPT (three), each blocked by a specific open item — dropping them would
# destroy the evidence needed to close it:
#   * n_cce_inmediatas, v_cce_inmediatas — O-3. The targets are labelled
#     "Interbancarias (Visa Direct)" but reports/02_correlation_findings.md
#     §5.3 describes CCE inmediatas as the rail both wallets ride. Both cannot
#     be true, and answering it needs these series.
#   * v_alias_intra_tot — O-5 (target units, miles vs millones). It is the
#     pre-split alias aggregate, so it should reconcile against the sum of the
#     wallet value series; that reconciliation IS the unit test. Coverage backs
#     this up: its last observation is 2026-06, matching the target family
#     rather than the 2026-07 of the true CCE series — it behaves like an alias
#     series, not a clearing-house one.
#
# Revisit once O-3 and O-5 close.
# --------------------------------------------------------------------------- #
CCE_SERIES = [
    {"code": "PN42233EM", "col_name": "v_cce_inmediatas",  "description": "CCE — Transferencias Inmediatas — Monto (millones S/)",               "category": "complementary", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42234EM", "col_name": "n_cce_inmediatas",  "description": "CCE — Transferencias Inmediatas — Número (miles)",                    "category": "complementary", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42661EM", "col_name": "v_alias_intra_tot", "description": "Pagos con alias — Valor Intrabancarias total (pre-wallet-split aggregate)", "category": "complementary", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
]

# Aggregate low-value payment instruments — BCRP table "Instrumentos de pagos de
# alto y bajo valor". These carry the long history the CCE codes were supposed
# to provide: monthly from Jan 2013, 162 observations confirmed by the coverage
# audit, zero interior gaps.
#
# Why these four:
#   * Dinero Electrónico is a SEPARATE instrument from Yape/Plin — prepaid
#     e-money balances, not bank-account transfers. In Jun 2026 it was ~19.7M
#     operations against ~1,114M intrabank transfers (about 1.8%). Useful as a
#     digital-adoption covariate; misleading if read as a substitute measure.
#   * Transferencias Intrabancarias is the aggregate PARENT of the main target:
#     n_transf_intra_yape is Yape's slice of exactly this flow. 2013–2026 gives
#     the pre-Yape baseline and the full adoption curve.
#
# Verified directly against the API: PN42180EM, PN42209EM, PN42200EM.
# PN42171EM is INFERRED from the monto/número offset of 29 that holds across the
# rest of the table — the fetch script will reject it loudly if wrong (O-4).
#
# Not pulled yet, one line away if the channel split proves interesting:
#   PN42172EM / PN42201EM  Intrabancarias — canales no presenciales
#   PN42173EM / PN42202EM  Intrabancarias — canales presenciales
PAGOS_AGREGADOS_SERIES = [
    {"code": "PN42209EM", "col_name": "n_dinero_electronico", "description": "Bajo valor — Dinero Electrónico — Número de operaciones (millones)",                        "category": "pagos_agregados", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42180EM", "col_name": "v_dinero_electronico", "description": "Bajo valor — Dinero Electrónico — Monto de operaciones (millones S/)",                      "category": "pagos_agregados", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42200EM", "col_name": "n_transf_intra_agg",   "description": "Bajo valor — Transferencias Intrabancarias (total sistema) — Número de operaciones (millones)", "category": "pagos_agregados", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
    {"code": "PN42171EM", "col_name": "v_transf_intra_agg",   "description": "Bajo valor — Transferencias Intrabancarias (total sistema) — Monto de operaciones (millones S/)", "category": "pagos_agregados", "frequency": "mensual", "kappa": KAPPA_PAYMENTS, "transform": "log_diff"},
]

# --------------------------------------------------------------------------- #
# Search interest -- Google Trends (plan 4.5, protocol 1.7).
#
# NOT a BCRP series: `source` says so, and fetch_target_series.py iterates
# BCRP_SERIES, so it never asks the BCRP API for this code. It lives in this
# registry anyway because the registry IS the single list of panel columns:
# raw.series_metadata is projected from ALL_SERIES, monthly_panel generates its
# columns from that table, and snapshot.py takes kappa/transform from here.
#
#   kappa = 0         month t is complete at the close of t and Trends updates
#                     within days -- two months fresher than the payments (4.1).
#   transform         log_diff: Trends rescales all history to the request's
#                     peak, and a log difference cancels that constant (4.2).
#   col_name          the series is yape + plin, in index points, from ONE
#                     request (one scale). The "code" is a label, not an API key.
# --------------------------------------------------------------------------- #
TRENDS_SERIES = [
    {"code": "GT_YAPE_PLIN", "col_name": "gt_yape_plin", "description": "Google Trends — Yape (Aplicación) + Plin (Tema), Perú, índice 0-100 (suma, una sola solicitud)", "category": "search_interest", "frequency": "mensual", "kappa": 0, "transform": "log_diff", "source": "google_trends"},
]

ALL_SERIES = TARGET_SERIES + MACRO_SERIES + CCE_SERIES + PAGOS_AGREGADOS_SERIES + TRENDS_SERIES

# What the BCRP fetch script may ask the API for. Everything without an explicit
# `source` is BCRP.
BCRP_SERIES = [s for s in ALL_SERIES if s.get("source", "bcrp") == "bcrp"]

# How far back to request each category, as BCRP period strings ("YYYY-M").
#
# The target series genuinely do not exist before Jan 2024 — the Yape/Plin
# breakdown starts there. Macro and CCE series were expected to have decades of
# history, and a single shared start date silently truncated them to the
# target's window.
#
# What the 2026-09-05 coverage audit actually found:
#   macro              2010-01 as requested, EXCEPT ingreso_formal, which
#                      starts 2015-01 at source. Harmless — w2019 needs history
#                      only back to 2018-01 — but worth knowing before anyone
#                      proposes a pre-2015 window.
#   complementary      2024-01 despite being requested from 2010-01. The API
#                      simply has nothing earlier. This is why the list shrank.
#   pagos_agregados    2013-01 as expected, 162 months.
#
# The API returns whatever exists, so an over-wide window is harmless.
START_BY_CATEGORY = {
    "target": "2024-1",
    "macro": "2010-1",
    "complementary": "2010-1",
    "pagos_agregados": "2013-1",
    # Not a BCRP request window: Google Trends is pulled by hand from 2017-01-01
    # (plan 4.5). Listed so the "category without a start date" guard
    # below stays a guard rather than learning an exception.
    "search_interest": "2017-1",
}

# Lookups keyed by series code — the single source of truth for renaming columns.
# Naming convention: n_ = número (count of operations), v_ = valor/monto (S/).
COL_NAME_BY_CODE = {s["code"]: s["col_name"] for s in ALL_SERIES}
DESCRIPTION_BY_CODE = {s["code"]: s["description"] for s in ALL_SERIES}

# Modelling lookups, keyed by panel column. snapshot.py writes these into the
# snapshot metadata CSV; dataset.py enforces them. Nothing in dbt reads them.
KAPPA_BY_COL = {s["col_name"]: s["kappa"] for s in ALL_SERIES}
TRANSFORM_BY_COL = {s["col_name"]: s["transform"] for s in ALL_SERIES}

# --------------------------------------------------------------------------- #
# Import-time guards. Every one of these exists because the failure it prevents
# would be SILENT: a wrong number in a feature matrix, not a crash.
# --------------------------------------------------------------------------- #

# Fail loudly if a col_name is ever duplicated — a silent collision would make
# two different series overwrite each other on pivot.
_dupes = {n for n in COL_NAME_BY_CODE.values() if list(COL_NAME_BY_CODE.values()).count(n) > 1}
if _dupes:
    raise ValueError(f"Duplicate col_name(s) in config: {sorted(_dupes)}")

# Same guard for series codes — a copy-paste slip would silently drop a series.
_codes = [s["code"] for s in ALL_SERIES]
_dupe_codes = {c for c in _codes if _codes.count(c) > 1}
if _dupe_codes:
    raise ValueError(f"Duplicate series code(s) in config: {sorted(_dupe_codes)}")

# And for categories: a typo would silently fall back to the wrong window.
_unknown = {s["category"] for s in ALL_SERIES} - set(START_BY_CATEGORY)
if _unknown:
    raise ValueError(f"No start date configured for category/ies: {sorted(_unknown)}")

# Every series needs a publication lag. A missing kappa would let dataset.py
# build a feature from a month that was not knowable at forecast time — a leak
# that produces excellent backtest numbers and a worthless model.
_missing_kappa = sorted(s["col_name"] for s in ALL_SERIES if s.get("kappa") is None)
if _missing_kappa:
    raise ValueError(f"Missing kappa (publication lag) for: {_missing_kappa}")

_bad_kappa = sorted(
    s["col_name"] for s in ALL_SERIES
    if not isinstance(s["kappa"], int) or isinstance(s["kappa"], bool) or s["kappa"] < 0
)
if _bad_kappa:
    raise ValueError(f"kappa must be a non-negative int; bad values for: {_bad_kappa}")

# Every series needs a transform. Defaulting to log_diff would quietly take the
# logarithm of a policy rate that can sit at zero.
_bad_transform = sorted(
    s["col_name"] for s in ALL_SERIES if s.get("transform") not in TRANSFORMS
)
if _bad_transform:
    raise ValueError(
        f"transform must be one of {sorted(TRANSFORMS)}; bad or missing for: {_bad_transform}"
    )
