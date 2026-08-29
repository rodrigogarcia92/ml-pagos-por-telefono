"""Series metadata for this project.

Mirrors the `raw.series_metadata` table defined in docs/data_sources.md —
keep the two in sync when series are added or the employment series (§7 of
that doc) is resolved.
"""

TARGET_SERIES = [
    {"code": "PN42672EM", "col_name": "n_transf_intra_yape", "description": "Número — Intrabancarias — Yape", "category": "target", "frequency": "mensual"},
    {"code": "PN42673EM", "col_name": "n_transf_intra_plin", "description": "Número — Intrabancarias — Plin", "category": "target", "frequency": "mensual"},
    {"code": "PN42677EM", "col_name": "n_transf_inter_visa_yape", "description": "Número — Interbancarias (Visa Direct) — Yape", "category": "target", "frequency": "mensual"},
    {"code": "PN42678EM", "col_name": "n_transf_inter_visa_plin", "description": "Número — Interbancarias (Visa Direct) — Plin", "category": "target", "frequency": "mensual"},
    {"code": "PN42662EM", "col_name": "v_transf_intra_yape", "description": "Valor (S/) — Intrabancarias — Yape", "category": "target", "frequency": "mensual"},
    {"code": "PN42663EM", "col_name": "v_transf_intra_plin", "description": "Valor (S/) — Intrabancarias — Plin", "category": "target", "frequency": "mensual"},
    {"code": "PN42667EM", "col_name": "v_transf_inter_visa_yape", "description": "Valor (S/) — Interbancarias — Yape", "category": "target", "frequency": "mensual"},
    {"code": "PN42668EM", "col_name": "v_transf_inter_visa_plin", "description": "Valor (S/) — Interbancarias — Plin", "category": "target", "frequency": "mensual"},
]

MACRO_SERIES = [
    {"code": "PN38705PM", "col_name": "ipc", "description": "Inflación (IPC)", "category": "macro", "frequency": "mensual"},
    {"code": "PN01246PM", "col_name": "tipo_cambio", "description": "Tipo de cambio nominal promedio (S/ por US$)", "category": "macro", "frequency": "mensual"},
    {"code": "PD04722MM", "col_name": "tasa_referencia", "description": "Tasa de Referencia de la Política Monetaria", "category": "macro", "frequency": "mensual"},
    {"code": "PN01770AM", "col_name": "pbi_idx", "description": "PBI mensual (índice 2007=100)", "category": "macro", "frequency": "mensual"},
    {"code": "PN00025MM", "col_name": "dolarizacion_liquidez", "description": "Coeficiente de Dolarización de la Liquidez (%)", "category": "macro", "frequency": "mensual"},
    {"code": "PN00048MM", "col_name": "circulante", "description": "Circulante — Emisión Primaria MN (millones S/)", "category": "macro", "frequency": "mensual"},
    {"code": "PN37696PM", "col_name": "ingreso_formal", "description": "Ingreso promedio sector formal privado — Nominal (S/)", "category": "macro", "frequency": "mensual"},
    # Empleo: TBD — see docs/data_sources.md §7 (BCRP series look thin, checking INEI ENAHO next)
]

# CCE aggregate series — broader payment-system context for Yape/Plin modelling.
# NOTE: these only reach back to 2024, so they do NOT provide the long history
# they were originally included for. That role now belongs to PAGOS_AGREGADOS
# below. Candidates for removal once the correlation work confirms they add
# nothing the aggregate series don't.
CCE_SERIES = [
    {"code": "PN42230EM", "col_name": "n_cce_cheques", "description": "CCE — Cheques — Número (miles)", "category": "complementary", "frequency": "mensual"},
    {"code": "PN42231EM", "col_name": "v_cce_credito", "description": "CCE — Transferencias de Crédito — Monto (millones S/)", "category": "complementary", "frequency": "mensual"},
    {"code": "PN42232EM", "col_name": "n_cce_credito", "description": "CCE — Transferencias de Crédito — Número (miles)", "category": "complementary", "frequency": "mensual"},
    {"code": "PN42233EM", "col_name": "v_cce_inmediatas", "description": "CCE — Transferencias Inmediatas — Monto (millones S/)", "category": "complementary", "frequency": "mensual"},
    {"code": "PN42234EM", "col_name": "n_cce_inmediatas", "description": "CCE — Transferencias Inmediatas — Número (miles)", "category": "complementary", "frequency": "mensual"},
    {"code": "PN42661EM", "col_name": "v_alias_intra_tot", "description": "Pagos con alias — Valor Intrabancarias total (pre-wallet-split aggregate)", "category": "complementary", "frequency": "mensual"},
]

# Aggregate low-value payment instruments — BCRP table "Instrumentos de pagos de
# alto y bajo valor". These carry the long history the CCE codes were supposed
# to provide: monthly from Jan 2013, ~163 observations.
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
# rest of the table — the fetch script will reject it loudly if wrong.
#
# Not pulled yet, one line away if the channel split proves interesting:
#   PN42172EM / PN42201EM  Intrabancarias — canales no presenciales
#   PN42173EM / PN42202EM  Intrabancarias — canales presenciales
PAGOS_AGREGADOS_SERIES = [
    {"code": "PN42209EM", "col_name": "n_dinero_electronico", "description": "Bajo valor — Dinero Electrónico — Número de operaciones (millones)", "category": "pagos_agregados", "frequency": "mensual"},
    {"code": "PN42180EM", "col_name": "v_dinero_electronico", "description": "Bajo valor — Dinero Electrónico — Monto de operaciones (millones S/)", "category": "pagos_agregados", "frequency": "mensual"},
    {"code": "PN42200EM", "col_name": "n_transf_intra_agg", "description": "Bajo valor — Transferencias Intrabancarias (total sistema) — Número de operaciones (millones)", "category": "pagos_agregados", "frequency": "mensual"},
    {"code": "PN42171EM", "col_name": "v_transf_intra_agg", "description": "Bajo valor — Transferencias Intrabancarias (total sistema) — Monto de operaciones (millones S/)", "category": "pagos_agregados", "frequency": "mensual"},
]

ALL_SERIES = TARGET_SERIES + MACRO_SERIES + CCE_SERIES + PAGOS_AGREGADOS_SERIES

# How far back to request each category, as BCRP period strings ("YYYY-M").
#
# The target series genuinely do not exist before Jan 2024 — the Yape/Plin
# breakdown starts there. Macro and CCE series have decades of history, and a
# single shared start date silently truncated them to the target's window,
# which removed the entire reason for including them: longer history for the
# panel model, and lagged macro features that predate the target.
#
# 2010 covers the full digital-payments era in Peru plus a long pre-Yape
# baseline, without dragging in structurally different pre-2000 regimes.
# The aggregate payment instruments start in 2013 at source, so asking from
# 2013 costs nothing. The API returns whatever exists, so an over-wide window
# is harmless.
START_BY_CATEGORY = {
    "target": "2024-1",
    "macro": "2010-1",
    "complementary": "2010-1",
    "pagos_agregados": "2013-1",
}

# Lookups keyed by series code — the single source of truth for renaming columns.
# Naming convention: n_ = número (count of operations), v_ = valor/monto (S/).
COL_NAME_BY_CODE = {s["code"]: s["col_name"] for s in ALL_SERIES}
DESCRIPTION_BY_CODE = {s["code"]: s["description"] for s in ALL_SERIES}

# Fail loudly at import time if a col_name is ever duplicated — a silent
# collision would make two different series overwrite each other on pivot.
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
