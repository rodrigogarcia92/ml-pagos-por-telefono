"""Series metadata for this project.

Mirrors the `raw.series_metadata` table defined in docs/data_sources.md —
keep the two in sync when series are added or the employment series (§6 of
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
    # Empleo: TBD — see docs/data_sources.md §6 (BCRP series look thin, checking INEI ENAHO next)
]

# CCE aggregate series — broader payment-system context for Yape/Plin modelling.
# Confirmed active from 2024 onwards; pre-2024 long-history codes still unresolved (open item).
CCE_SERIES = [
    {"code": "PN42230EM", "col_name": "n_cce_cheques", "description": "CCE — Cheques — Número (miles)", "category": "complementary", "frequency": "mensual"},
    {"code": "PN42231EM", "col_name": "v_cce_credito", "description": "CCE — Transferencias de Crédito — Monto (millones S/)", "category": "complementary", "frequency": "mensual"},
    {"code": "PN42232EM", "col_name": "n_cce_credito", "description": "CCE — Transferencias de Crédito — Número (miles)", "category": "complementary", "frequency": "mensual"},
    {"code": "PN42233EM", "col_name": "v_cce_inmediatas", "description": "CCE — Transferencias Inmediatas — Monto (millones S/)", "category": "complementary", "frequency": "mensual"},
    {"code": "PN42234EM", "col_name": "n_cce_inmediatas", "description": "CCE — Transferencias Inmediatas — Número (miles)", "category": "complementary", "frequency": "mensual"},
    {"code": "PN42661EM", "col_name": "v_alias_intra_tot", "description": "Pagos con alias — Valor Intrabancarias total (pre-wallet-split aggregate)", "category": "complementary", "frequency": "mensual"},
]

ALL_SERIES = TARGET_SERIES + MACRO_SERIES + CCE_SERIES

# Lookups keyed by series code — the single source of truth for renaming columns.
# Naming convention: n_ = número (count of operations), v_ = valor/monto (S/).
COL_NAME_BY_CODE = {s["code"]: s["col_name"] for s in ALL_SERIES}
DESCRIPTION_BY_CODE = {s["code"]: s["description"] for s in ALL_SERIES}

# Fail loudly at import time if a col_name is ever duplicated — a silent
# collision would make two different series overwrite each other on pivot.
_dupes = {n for n in COL_NAME_BY_CODE.values() if list(COL_NAME_BY_CODE.values()).count(n) > 1}
if _dupes:
    raise ValueError(f"Duplicate col_name(s) in config: {sorted(_dupes)}")
