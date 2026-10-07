# %% [markdown]
# # EDA — BCRP Raw Data
#
# Reads the raw JSON snapshots in `data/raw/bcrp/` and builds two DataFrames:
#
# * `df`      — long format, one row per (series, month, pull). Good for filtering/grouping.
# * `df_wide` — wide format, one row per month, one column per series, keeping only
#               the most recent pull per (series, month). Good for correlations,
#               plotting, and eventually modelling.
#
# Parsing lives in `src/data_collection/parse.py` so this notebook and the
# BigQuery loader share one parser and can never drift apart.
#
# Run cells one at a time in VS Code (Shift+Enter), or the whole file:
#   python -m notebooks.01_eda_bcrp

# %% — Imports & paths
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# Make `src` importable whether this runs as a script or cell-by-cell.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.config import COL_NAME_BY_CODE, DESCRIPTION_BY_CODE  # noqa: E402
from src.data_collection.parse import load_all_snapshots, to_wide  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


# %% — Load every snapshot
raw_all = load_all_snapshots()

print(f"\nRows loaded       : {len(raw_all):,}")
print(f"Pull dates present: {sorted(raw_all['pulled_at'].dt.date.unique())}")


# %% — Keep only the most recent pull per (series, month)
# BCRP revises recent months, so the newest pull wins by design.
df = (
    raw_all.sort_values(["series_code", "obs_date", "pulled_at"])
    .drop_duplicates(subset=["series_code", "obs_date"], keep="last")
    .reset_index(drop=True)
)

n_dropped = len(raw_all) - len(df)
print(f"Rows after de-duplication : {len(df):,}  ({n_dropped:,} superseded rows dropped)")
print(f"Series                    : {df['series_code'].nunique()}")
print(f"Date range                : {df['obs_date'].min():%b %Y} → {df['obs_date'].max():%b %Y}")


# %% — Coverage check: how much history does each series actually have?
coverage = (
    df.groupby(["col_name", "series_code"])
    .agg(
        first_obs=("obs_date", "min"),
        last_obs=("obs_date", "max"),
        n_obs=("value", "count"),
        n_missing=("value", lambda s: s.isna().sum()),
    )
    .reset_index()
    .sort_values("col_name")
)
coverage["first_obs"] = coverage["first_obs"].dt.strftime("%b %Y")
coverage["last_obs"] = coverage["last_obs"].dt.strftime("%b %Y")

print("--- Coverage by series ---")
print(coverage.to_string(index=False))

empty = coverage[coverage["n_obs"] == 0]
if not empty.empty:
    print("\n⚠ Series with ZERO observations — check the code is still active:")
    print(empty[["series_code", "col_name"]].to_string(index=False))
else:
    print("\n✓ Every series returned data.")


# %% — Pivot to wide: one row per month, one column per series
df_wide = to_wide(raw_all)

print(f"df_wide shape: {df_wide.shape[0]} months × {df_wide.shape[1]} series\n")
print(df_wide.dtypes.to_string())


# %% — First and last few months
print("--- Head ---")
print(df_wide.head(3).to_string())
print("\n--- Tail ---")
print(df_wide.tail(3).to_string())


# %% — Missing values per column (tail months often lag by series)
missing = pd.DataFrame({
    "n_missing": df_wide.isna().sum(),
    "pct_missing": (df_wide.isna().mean() * 100).round(1),
})
print("--- Missing values ---")
print(missing[missing["n_missing"] > 0].to_string() if missing["n_missing"].any()
      else "None — every series is complete across all months.")


# %% — Descriptive statistics
print("--- Descriptive stats ---")
print(df_wide.describe().T.round(2).to_string())


# %% — Series reference: column name → what it actually is
reference = pd.DataFrame(
    [(code, COL_NAME_BY_CODE[code], DESCRIPTION_BY_CODE[code]) for code in sorted(COL_NAME_BY_CODE)],
    columns=["series_code", "col_name", "description"],
)
print("--- Series reference ---")
print(reference.to_string(index=False))


# %% [markdown]
# ## Explore from here
#
# `df_wide` is indexed by month, one column per series. Some starting points:
#
# ```python
# # The four Yape/Plin volume series
# df_wide.filter(like="_yape")
# df_wide[["n_transf_intra_yape", "n_transf_intra_plin"]].plot()
#
# # Month-over-month growth
# df_wide["n_transf_intra_yape"].pct_change().mul(100).round(1)
#
# # Yape share of intrabank alias payments
# (df_wide["v_transf_intra_yape"] / df_wide["v_alias_intra_tot"]).mul(100).round(1)
#
# # Correlation among the target series
# df_wide.filter(regex="^n_transf").corr().round(2)
#
# # Long format is easier for grouped questions
# df[df["col_name"].str.startswith("v_")].groupby("col_name")["value"].mean()
# ```

# %%
df_wide["n_transf_total"] = df_wide[[
    "n_transf_intra_yape", "n_transf_intra_plin",
    "n_transf_inter_visa_yape", "n_transf_inter_visa_plin",
]].sum(axis=1, min_count=4)

df_wide["v_transf_digital"] = df_wide[[
    "v_transf_intra_yape", "v_transf_intra_plin",
    "v_transf_inter_visa_yape", "v_transf_inter_visa_plin",
]].sum(axis=1, min_count=4)

# %%
# Month-over-month growth
df_wide["v_transf_digital"].pct_change().mul(100).round(1)
(df_wide["v_transf_intra_yape"] / df_wide["v_alias_intra_tot"]).mul(100).round(1)

# Correlation among the target series
df_wide.filter(regex="^n_transf").corr().round(2)

# %%
def plot_series_with_trend(
    df,
    column,
    title=None,
    ylabel=None,
    frac=0.4,
    ax=None,
    style=True,
):
    """Plot one series from a wide DataFrame with a LOWESS trend line.

    Parameters
    ----------
    df : DataFrame with a DatetimeIndex, one column per series.
    column : str, column to plot.
    frac : float, fraction of points in each local LOWESS fit. With ~30
        monthly observations, 0.3-0.5 shows a trend; 0.1 gives ~3 points
        per fit and simply interpolates the noise.
    ax : optional existing Axes, for placing this in a grid of panels.
    style : apply seaborn styling. Set False to avoid changing global
        rcParams, e.g. when composing with other plots.

    Returns the Axes, so you can adjust it after the call.
    """
    if column not in df.columns:
        raise KeyError(
            f"{column!r} not in DataFrame. Available: {sorted(df.columns)}"
        )

    # Drop missing months explicitly. LOWESS drops them silently, which
    # makes a short series look complete when it isn't.
    s = df[column].dropna()
    n_missing = df[column].isna().sum()
    if len(s) < 4:
        raise ValueError(f"{column!r}: only {len(s)} non-null points, too few to smooth.")

    if style:
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=1.2)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        s.index, s.values,
        marker="o", markersize=5, linestyle="-", linewidth=1.5,
        color="#1f77b4", label="Actual", alpha=0.8,
    )

    # date2num handles any datetime resolution. Converting via
    # astype(int64) assumes nanoseconds and breaks on pandas 3.x, where
    # the default is datetime64[us] -- the trend line lands in 1970.
    x_num = mdates.date2num(s.index.to_pydatetime())
    smooth = lowess(s.values, x_num, frac=frac, return_sorted=True)

    ax.plot(
        mdates.num2date(smooth[:, 0]), smooth[:, 1],
        color="red", linewidth=3,
        label=f"Trend (LOWESS, frac={frac})",
    )

    ax.set_title(title or column, fontsize=18, weight="bold")
    ax.set_xlabel("Date", fontsize=15)
    ax.set_ylabel(ylabel or column, fontsize=15)
    ax.legend(frameon=True, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)

    if n_missing:
        ax.text(
            0.99, 0.01, f"{n_missing} month(s) missing",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, style="italic", alpha=0.7,
        )

    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    return ax


# %%
plot_series_with_trend(
    df_wide, "n_transf_total",
    title="Número de operaciones de transferencia digital inmediata (Yape y Plin)",
    ylabel="Millones de operaciones",
)
plt.tight_layout()
plt.show()
# %%
