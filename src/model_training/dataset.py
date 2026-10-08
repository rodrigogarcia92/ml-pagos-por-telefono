"""Snapshot -> (X, y, ctx). The single place features are built.

Every model in the grid is evaluated on identical inputs because they all come
through here. The three rules this module enforces, each of which is a silent
bug if it is wrong (plan 1.3, 1.4, 4.2):

  1. NON-ANTICIPATION. A feature from series j may only reference month
     t - k with k >= kappa_j. Enforced against the snapshot metadata, not
     against a hand-maintained lag list. A violation RAISES.
  2. TWO INDEXING RULES. Stochastic features (macro, own history) are indexed
     at t - k. Deterministic features (calendar, COVID dummies) are indexed at
     the TARGET month t - kappa + h, because they are known for any future
     month. Using the origin month would be wrong by h - kappa months.
  3. TRANSFORM PER SERIES. Counts and levels take log differences; percents and
     shares take simple differences in percentage points. A log difference of a
     policy rate that can sit at zero is undefined.

Row index is the ORIGIN month t: the month at whose close the forecast is made.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Panel column -> feature-name prefix, per plan 4.4.
PREFIX = {
    "circulante": "circ",
    "pbi_idx": "pbi",
    "ipc": "ipc",
    "tipo_cambio": "tc",
    "ingreso_formal": "ing",
    "tasa_referencia": "tasa",
    "dolarizacion_liquidez": "dol",
    # Google Trends (plan 4.5, protocol 1.7). kappa = 0, so gt_d0 -- the
    # change in the latest complete month -- is admissible. The names follow the
    # convention below, NOT the owner's "gt_d1"/"gt_d12": here `_d{k}` is the one-month
    # change at lag k, so gt_d1 would be last month's change and gt_d12 the change twelve
    # months ago. The year-on-year idea is gt_ma12 (12-month mean change = yoy / 12).
    "gt_yape_plin": "gt",
}

# Target definitions. A target is one panel column or the sum of several.
TARGETS = {
    "t2": {"cols": ["n_transf_intra_agg"], "horizon": 1},
    "t3": {"cols": ["n_transf_intra_agg"], "horizon": 3},
    "t4": {"cols": ["n_transf_intra_yape", "n_transf_intra_plin"], "horizon": 1},
    "t5": {"cols": ["n_transf_intra_yape", "n_transf_intra_plin"], "horizon": 3},
    # Protocol 1.7 (plan 3, Targets 10-11): one quarter ahead of the forecast date.
    # With kappa = 2 the target month is t - kappa + h = t + 3 (h=5) / t + 4 (h=6),
    # z5 = log n_{t+3} - log n_{t-2}; purge h - 1 = 4 / 5. IDs t6-t9 stay reserved for
    # the deferred value/share targets.
    "t10": {"cols": ["n_transf_intra_agg"], "horizon": 5},
    "t11": {"cols": ["n_transf_intra_agg"], "horizon": 6},
    # t1 (baselines) reuses t2/t3's series -- the floor must be measured on the
    # same target it is a floor for.
    "t1": {"cols": ["n_transf_intra_agg"], "horizon": 1},
}

# First TARGET month of each estimation window (protocol 1.6, O-10). There is no
# end date on purpose: the last row is the last month whose target has been
# published, i.e. whatever the `valid` mask leaves in build(). The origin range
# is derived from it: origin = target - (horizon - kappa) months, so the same
# window covers the same calendar target months at h=1 and h=3. Bounding the
# ORIGIN instead (protocols 1.2-1.5) left the newest published month unused as a
# target and shifted the final holdout by a different amount at each horizon.
WINDOWS = {
    "w2019": "2019-01-01",
    "w2021": "2021-01-01",
    "w2024": "2024-01-01",
}

# Seasonal-transfer factors (plan 5.2, pre-registered 2026-10-05).
# Month-of-year mean of the t2 target (dlog of the aggregate) over TARGET months
# first <= tau < cutoff, minus the COVID pulse months, demeaned to sum to zero.
# Measured on the aggregate -- never on a fitted model, and never on any month
# the wallet windows forecast, which is what makes it leak-free for every fold.
# Changing any of these four is a protocol change (1.7), not an edit.
SEAS_TRANSFER_SOURCE = "n_transf_intra_agg"
SEAS_TRANSFER_FIRST_TARGET = pd.Timestamp("2019-01-01")
SEAS_TRANSFER_CUTOFF = pd.Timestamp("2024-01-01")           # exclusive
SEAS_TRANSFER_EXCLUDE = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-09-01"))  # inclusive

# Peru national holidays with fixed dates. Movable feasts (Jueves and Viernes
# Santo) are derived from Easter below. Aug 6 (Batalla de Junin) became a
# national holiday in 2024 and is handled by the year guard.
FIXED_HOLIDAYS = [
    (1, 1), (5, 1), (6, 29), (7, 28), (7, 29), (8, 30),
    (10, 8), (11, 1), (12, 8), (12, 9), (12, 25),
]


def _easter(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm. Avoids a dependency for fifteen lines."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = ((h + m - 7 * n + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def _holidays_in_month(ts: pd.Timestamp) -> int:
    year, month = ts.year, ts.month
    n = sum(1 for (mm, _) in FIXED_HOLIDAYS if mm == month)
    if month == 8 and year >= 2024:
        n += 1  # Batalla de Junin
    easter = _easter(year)
    for offset in (-3, -2):  # Jueves Santo, Viernes Santo
        d = easter + pd.Timedelta(days=offset)
        if d.month == month:
            n += 1
    return n


def _calendar_frame(months: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic features for the months given. Caller passes TARGET months."""
    rows = []
    for m in months:
        days = calendar.monthrange(m.year, m.month)[1]
        weekend = sum(
            1 for d in range(1, days + 1)
            if pd.Timestamp(year=m.year, month=m.month, day=d).weekday() >= 5
        )
        rows.append({
            "cal_days": days,
            "cal_weekend_days": weekend,
            "cal_holidays": _holidays_in_month(m),
            "cal_month": m.month,
            # A level shock appears as a PULSE in a differenced target, not as a
            # sustained block -- two parameters rather than six (4.4).
            "d_covid_collapse": int((m.year == 2020) and (m.month in (3, 4))),
            "d_covid_rebound": int((m.year == 2020) and (5 <= m.month <= 9)),
        })
    return pd.DataFrame(rows, index=months)


def seasonal_transfer_factors(panel: pd.DataFrame) -> pd.Series:
    """The twelve month-of-year factors, index 1..12, summing to zero (plan 5.2).

    Reads ONLY target months in [SEAS_TRANSFER_FIRST_TARGET, SEAS_TRANSFER_CUTOFF)
    outside the COVID pulse, so nothing at or after the cutoff can move it --
    pinned by test_seas_transfer_factors_ignore_everything_from_the_cutoff.
    """
    dlog = np.log(panel[SEAS_TRANSFER_SOURCE]).diff()     # indexed at the target month
    tau = dlog.index
    covid = (tau >= SEAS_TRANSFER_EXCLUDE[0]) & (tau <= SEAS_TRANSFER_EXCLUDE[1])
    use = (tau >= SEAS_TRANSFER_FIRST_TARGET) & (tau < SEAS_TRANSFER_CUTOFF) & ~covid
    d = dlog[use]
    if d.isna().any():
        raise ValueError(f"{SEAS_TRANSFER_SOURCE} has gaps in the seasonal-transfer sample")
    by_month = d.groupby(d.index.month).mean()
    if len(by_month) != 12:
        raise ValueError(f"seasonal-transfer sample covers {len(by_month)} calendar months, not 12")
    return by_month - by_month.mean()


def _diff(series: pd.Series, transform: str) -> pd.Series:
    if transform == "log_diff":
        return np.log(series).diff()
    if transform == "simple_diff":
        return series.diff()
    raise ValueError(f"Unknown transform {transform!r}")


@dataclass
class Frame:
    """One (target, horizon, window, feature set, encoding) design matrix.

    ctx carries what the naive baselines and the metrics need but the models
    must not see as features: the anchor level, the realised level, and the
    seasonal reference. Keeping them out of X is what stops a model reading the
    answer off its own input.
    """
    X: pd.DataFrame
    y: pd.Series           # z_h, the differenced target the models are fitted on
    ctx: pd.DataFrame      # anchor, y_level, seas_level, cal_days_m, cal_days_m12
    horizon: int
    window: str
    feature_set: str
    encoding: str
    dropped: list[str]     # columns removed as all-null or zero-variance
    target_start: pd.Timestamp | None = None   # first TARGET month in the frame
    target_end: pd.Timestamp | None = None     # last TARGET month in the frame
    # Provenance of `seas_transfer` when it is a column (else None): where the
    # factors came from, the cutoff, and the twelve values. Logged by train.py.
    seas_transfer: dict | None = None
    # The SAME rows in the other encodings, for a model family that mixes them (ens3:
    # SVR one-hot, trees integer). Identical index, target and ctx by construction;
    # train.fit_config asserts it. Empty for every other family.
    alt_encodings: dict | None = None

    def for_encoding(self, encoding: str) -> Frame:
        """This frame's design matrix in `encoding` (itself, or the stored alternative)."""
        if encoding == self.encoding:
            return self
        if not self.alt_encodings or encoding not in self.alt_encodings:
            raise KeyError(f"frame has no {encoding!r} encoding (has {self.encoding!r})")
        return self.alt_encodings[encoding]


def load_snapshot(panel_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_path = Path(panel_path)
    version = panel_path.stem.removeprefix("panel_")
    meta_path = panel_path.with_name(f"series_meta_{version}.csv")
    panel = pd.read_parquet(panel_path)
    panel.index = pd.to_datetime(panel.index)
    meta = pd.read_csv(meta_path).set_index("col_name")
    return panel.sort_index(), meta


def build(
    panel: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    target_id: str,
    horizon: int,
    window: str,
    columns: list[str],
    encoding: str = "int",
    keep_unlabelled: bool = False,
) -> Frame:
    """The design matrix. `keep_unlabelled=True` is the opt-in used ONLY by
    build_prediction_rows: it also keeps rows whose target has not been published yet
    (y and y_level NaN). The default path -- everything training, tuning and evaluation
    see -- is exactly what it has always been."""
    spec = TARGETS[target_id]
    y_level = panel[spec["cols"]].sum(axis=1, min_count=len(spec["cols"]))
    kappa_y = int(meta.loc[spec["cols"][0], "kappa"])

    # --- differenced series, built ON DEMAND -----------------------------------
    # Lazily, not for all 22 columns up front. Two reasons, one of them a real
    # bug: n_dinero_electronico is exactly 0 for its first months in 2013, so
    # eagerly taking log() of every column emitted
    #     RuntimeWarning: divide by zero encountered in log
    # on every single build -- for a column no feature set even uses. The
    # -inf never reached a model, but a warning you learn to ignore is a
    # warning that will hide a real one later. Also ~10x less work per call.
    _cache: dict[str, pd.Series] = {"__y__": _diff(y_level, "log_diff")}

    def diff_of(src: str) -> pd.Series:
        if src not in _cache:
            if src not in panel.columns:
                raise KeyError(
                    f"{src} is not in this snapshot. Google Trends needs a snapshot taken "
                    "after load_trends + dbt build + snapshot (plan 9.2 step 3)."
                    if src == "gt_yape_plin" else f"{src} is not in this snapshot."
                )
            s = panel[src]
            if meta.loc[src, "transform"] == "log_diff" and (s <= 0).any():
                hint = (
                    " Trends printed a bare 0, not '<1' (which the plan reads as 0.5, 4.5): "
                    "that is a data-gate finding (G1), and nothing here floors or smooths it."
                    if src == "gt_yape_plin" else ""
                )
                raise ValueError(
                    f"{src} contains non-positive values, so log_diff is undefined. "
                    f"First offender: {s[s <= 0].index[0]:%Y-%m}. Fix the transform "
                    f"in config.py, or exclude the series.{hint}"
                )
            _cache[src] = _diff(s, meta.loc[src, "transform"])
        return _cache[src]

    idx = panel.index

    # TARGET MONTH = t - kappa + h.
    #
    # The plan defines h as "steps beyond the LAST OBSERVATION", and the last
    # observation is t - kappa. With kappa=1 that reduces to the familiar
    # t + h - 1; with kappa=2 (payments, confirmed 2026-09-05) it is t + h - 2.
    #
    # Hardcoding `horizon - 1` would have silently kept the old convention when
    # kappa changed, making z span h-1 months while every drift model still
    # divided by h. Deriving it from kappa keeps z an exact h-step difference
    # whatever the release calendar does next.
    target_months = idx + pd.DateOffset(months=horizon - kappa_y)

    # --- target: z_h = log n_{t-kappa+h} - log n_{t-kappa} -------------------
    # Anchor and target are exactly h months apart, by construction.
    anchor = y_level.shift(kappa_y)                       # n_{t-kappa}
    realised = y_level.reindex(target_months).to_numpy()  # n_{t-kappa+h}
    realised = pd.Series(realised, index=idx)
    z = np.log(realised) - np.log(anchor)

    # --- deterministic block, indexed at the TARGET month --------------------
    cal = _calendar_frame(pd.DatetimeIndex(target_months))
    cal.index = idx

    seas_months = target_months - pd.DateOffset(months=12)
    ctx = pd.DataFrame({
        "anchor": anchor,
        "y_level": realised,
        "seas_level": pd.Series(y_level.reindex(seas_months).to_numpy(), index=idx),
        "cal_days_m": cal["cal_days"].to_numpy(),
        "cal_days_m12": [
            calendar.monthrange(m.year, m.month)[1] for m in pd.DatetimeIndex(seas_months)
        ],
    }, index=idx)

    # --- assemble requested columns -----------------------------------------
    feats: dict[str, pd.Series] = {}
    transfer: dict | None = None
    for name in columns:
        if name in cal.columns:
            feats[name] = cal[name]
            continue
        if name == "seas_transfer":
            # Deterministic, so indexed at the TARGET month like the calendar block.
            # THE GUARD: the factors are measured on target months before the
            # cutoff, so a window that starts earlier would be scoring its own
            # training data with them. Refuse rather than leak.
            if pd.Timestamp(WINDOWS[window]) < SEAS_TRANSFER_CUTOFF:
                raise ValueError(
                    f"seas_transfer is only admissible on windows starting at or after "
                    f"{SEAS_TRANSFER_CUTOFF:%Y-%m}; {window} starts {WINDOWS[window][:7]} "
                    "and would score months its own factors were measured on."
                )
            factors = seasonal_transfer_factors(panel)
            feats[name] = pd.Series(
                factors.reindex(pd.DatetimeIndex(target_months).month).to_numpy(), index=idx
            )
            transfer = {
                "source": SEAS_TRANSFER_SOURCE,
                "first_target": f"{SEAS_TRANSFER_FIRST_TARGET:%Y-%m}",
                "cutoff": f"{SEAS_TRANSFER_CUTOFF:%Y-%m}",
                "excludes": f"{SEAS_TRANSFER_EXCLUDE[0]:%Y-%m}..{SEAS_TRANSFER_EXCLUDE[1]:%Y-%m}",
                "factors": {int(m): float(v) for m, v in factors.items()},
            }
            continue
        if name.endswith("_lvl"):                       # FS5b level variants
            src = name.removesuffix("_lvl")
            src = {"dolarizacion": "dolarizacion_liquidez"}.get(src, src)
            k = int(meta.loc[src, "kappa"])
            feats[name] = panel[src].shift(k)
            continue

        prefix, _, suffix = name.rpartition("_")
        src = "__y__" if prefix == "y" else next(
            (c for c, p in PREFIX.items() if p == prefix), None
        )
        if src is None:
            raise KeyError(f"No panel series maps to feature prefix {prefix!r} (column {name!r})")
        kappa = kappa_y if src == "__y__" else int(meta.loc[src, "kappa"])
        d = diff_of(src)

        if suffix.startswith("d"):
            k = int(suffix[1:])
            # THE GUARD. A feature at t-k with k < kappa_j would use a figure
            # that had not been published at forecast time. Excellent backtests,
            # worthless model.
            if k < kappa:
                raise ValueError(
                    f"{name}: lag {k} violates kappa={kappa} for {src}. "
                    "Fix the feature set, or fix kappa in config.py -- and log "
                    "the change in plan 11."
                )
            feats[name] = d.shift(k)
        elif suffix.startswith("ma"):
            w = int(suffix[2:])
            # Anchored at the most recent ADMISSIBLE observation, not at t.
            feats[name] = d.shift(kappa).rolling(w).mean()
        else:
            raise KeyError(f"Cannot parse feature name {name!r}")

    X = pd.DataFrame(feats, index=idx)

    if encoding == "onehot" and "cal_month" in X.columns:
        dummies = pd.get_dummies(X["cal_month"], prefix="cal_m", drop_first=True).astype(int)
        X = X.drop(columns="cal_month").join(dummies)
    elif encoding not in ("int", "onehot"):
        raise ValueError(f"encoding must be 'int' or 'onehot', got {encoding!r}")

    # --- window, then drop rows that cannot be complete ----------------------
    # The window bounds the TARGET month (O-10). Feature history is still read
    # from the full panel above, so lags may look back before the window start.
    keep = np.asarray(target_months >= pd.Timestamp(WINDOWS[window]))
    X, z, ctx = X.loc[keep], z.loc[keep], ctx.loc[keep]

    if keep_unlabelled:
        # Prediction rows: every feature and the anchor must exist (the kappa guard above has
        # already run); only the realised target may be missing.
        valid = X.notna().all(axis=1) & ctx["anchor"].notna()
    else:
        valid = X.notna().all(axis=1) & z.notna() & ctx[["anchor", "y_level"]].notna().all(axis=1)
    X, z, ctx = X.loc[valid], z.loc[valid], ctx.loc[valid]

    # --- per-window column hygiene -------------------------------------------
    # COVID dummies are identically zero in w2021 and w2024; a constant column
    # is noise for a tree and a singularity for a linear model. Dropped here and
    # logged, so "which columns did this run actually use" is answerable later.
    dropped = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=dropped)

    tgt = X.index + pd.DateOffset(months=horizon - kappa_y)
    return Frame(
        X=X, y=z, ctx=ctx, horizon=horizon, window=window,
        feature_set="", encoding=encoding, dropped=dropped,
        target_start=tgt.min() if len(tgt) else None,
        target_end=tgt.max() if len(tgt) else None,
        seas_transfer=transfer,
    )


def build_prediction_rows(
    panel: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    target_id: str,
    horizon: int,
    window: str,
    columns: list[str],
    encoding: str = "int",
    extend_months: int = 6,
) -> Frame:
    """The origins that can be forecast but cannot yet be scored: features complete under their
    kappa, anchor published, target month not yet published.

    The row index is the origin month, as everywhere in this module, and `y` / `ctx["y_level"]`
    are NaN. Columns are exactly those of the ordinary `build` (so a model fitted on the
    labelled frame can be applied as is), including any zero-variance column it dropped.

    The panel is extended with empty future months first: its index may stop before the latest
    admissible origin (the newest month of a fast series can sit AHEAD of the newest payments
    month), and lags are positional. Empty months carry no data, so nothing is invented.

    The training path is untouched -- `build` is called with its default flag for the labelled
    frame, and the κ guard runs in both calls.
    """
    labelled = build(panel, meta, target_id=target_id, horizon=horizon, window=window,
                     columns=columns, encoding=encoding)
    full = pd.date_range(panel.index.min(), panel.index.max() + pd.DateOffset(months=extend_months),
                         freq="MS")
    ext = build(panel.reindex(full), meta, target_id=target_id, horizon=horizon, window=window,
                columns=columns, encoding=encoding, keep_unlabelled=True)

    pending = ext.y.isna() | ext.ctx["y_level"].isna()
    missing = [c for c in labelled.X.columns if c not in ext.X.columns]
    if missing:
        raise RuntimeError(f"prediction rows lack the training columns {missing}")
    if not labelled.X.index.isin(ext.X.index[~pending]).all():
        raise RuntimeError("prediction build disagrees with the training build on labelled rows")
    X = ext.X.loc[pending, list(labelled.X.columns)]
    return Frame(
        X=X, y=ext.y.loc[pending], ctx=ext.ctx.loc[pending], horizon=horizon, window=window,
        feature_set="", encoding=encoding, dropped=labelled.dropped,
        target_start=ext.target_start, target_end=ext.target_end, seas_transfer=ext.seas_transfer,
    )
