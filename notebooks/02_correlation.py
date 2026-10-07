# %% [markdown]
# # 02 — Yape/Plin vs aggregate digital-payment series
#
# Reads `marts.monthly_panel` from BigQuery and asks one question:
#
# > Do the long-history aggregate payment series move with Yape + Plin, and
# > would they be useful features or panel-model companions?
#
# **The methodological point of this notebook.** Two series that both trend
# upward will correlate at ~0.99 whether or not they have anything to do with
# each other. That is a property of the trend, not of the relationship. So we
# compute correlations twice:
#
# * on **levels** — reported only to show how misleading they are;
# * on **month-over-month log growth** — the honest measure, and the one to act on.
#
# Run cells one at a time in VS Code (Shift+Enter), or the whole file:
#   python -m notebooks.02_correlation
#
# Requires an up-to-date warehouse:
#   python -m src.data_collection.fetch_target_series
#   python -m src.data_collection.load_to_bigquery
#   (then, in .venv-dbt)  dbt build

# %% — Imports & connection
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from google.cloud import bigquery

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
PROJECT_ID = os.environ["GCP_PROJECT_ID"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

# Okabe–Ito, ordered so no adjacent pair is hard to separate under deuteranopia.
# Colourblind-safe by construction; the ordering was checked rather than assumed.
PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
INK = "#16202B"
GRID = "#D5DBE2"


# %% — Load the mart
client = bigquery.Client(project=PROJECT_ID)

panel = (
    client.query(f"SELECT * FROM `{PROJECT_ID}.marts.monthly_panel` ORDER BY obs_date")
    .to_dataframe()
    .set_index("obs_date")
)
panel.index = pd.to_datetime(panel.index)

print(f"monthly_panel: {panel.shape[0]} months × {panel.shape[1]} series")
print(f"Range: {panel.index.min():%b %Y} → {panel.index.max():%b %Y}")


# %% — Which columns are we working with?
TARGETS = [
    "n_transf_intra_yape", "n_transf_intra_plin",
    "n_transf_inter_visa_yape", "n_transf_inter_visa_plin",
]
AGGREGATES = [
    "n_transf_intra_agg", "v_transf_intra_agg",
    "n_dinero_electronico", "v_dinero_electronico",
]

missing = [c for c in TARGETS + AGGREGATES if c not in panel.columns]
if missing:
    raise KeyError(
        f"Not in monthly_panel: {missing}\n"
        "Re-run the fetch script, the loader, and `dbt build`. If only "
        "v_transf_intra_agg is missing, PN42171EM (an inferred code) was "
        "rejected by the API — drop it from config.py."
    )

# Combined wallet volume: the thing we actually care about explaining.
panel["n_yape_plin"] = panel[TARGETS].sum(axis=1, min_count=len(TARGETS))
panel["v_yape_plin"] = panel[
    ["v_transf_intra_yape", "v_transf_intra_plin",
     "v_transf_inter_visa_yape", "v_transf_inter_visa_plin"]
].sum(axis=1, min_count=4)


# %% — Coverage: how much history does each series really have?
cols = ["n_yape_plin", "v_yape_plin"] + AGGREGATES
coverage = pd.DataFrame({
    "first_obs": [panel[c].first_valid_index() for c in cols],
    "last_obs": [panel[c].last_valid_index() for c in cols],
    "n_obs": [panel[c].notna().sum() for c in cols],
}, index=cols)
coverage["first_obs"] = coverage["first_obs"].dt.strftime("%b %Y")
coverage["last_obs"] = coverage["last_obs"].dt.strftime("%b %Y")

print("--- Coverage ---")
print(coverage.to_string())
print(
    "\nThe aggregates reach back to 2013; the wallet split starts in 2024.\n"
    "Any correlation BETWEEN the two groups is therefore limited to the\n"
    "overlap window — roughly 30 monthly observations. Small."
)


# %% — Long-history context (small multiples, NOT a dual axis)
#
# These two series differ by ~50x in magnitude. Plotting them on twin y-axes
# would let the visual slope be set by an arbitrary scaling choice rather than
# by the data. Separate panels sharing an x-axis keep every comparison honest.

hist = panel.loc["2013":, ["n_transf_intra_agg", "n_dinero_electronico"]].dropna(how="all")
wallet_start = panel["n_yape_plin"].first_valid_index()

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
specs = [
    ("n_transf_intra_agg", "Transferencias intrabancarias (sistema)", PALETTE[0]),
    ("n_dinero_electronico", "Dinero electrónico", PALETTE[1]),
]

for ax, (col, label, color) in zip(axes, specs, strict=False):
    ax.plot(hist.index, hist[col], color=color, linewidth=2)
    ax.set_ylabel("Millones de operaciones", fontsize=10, color=INK)
    ax.set_title(label, fontsize=12, loc="left", color=INK, pad=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)

    if wallet_start is not None:
        ax.axvline(wallet_start, color=INK, linestyle="--", linewidth=1, alpha=0.45)

    # Direct label at the final point — no legend box needed for one series.
    last = hist[col].dropna()
    ax.annotate(
        f"{last.iloc[-1]:,.1f}",
        xy=(last.index[-1], last.iloc[-1]),
        xytext=(6, 0), textcoords="offset points",
        va="center", fontsize=10, color=INK,
    )

axes[0].annotate(
    "Yape/Plin split\nbegins",
    xy=(wallet_start, axes[0].get_ylim()[1] * 0.72),
    xytext=(-84, 0), textcoords="offset points",
    fontsize=9, color=INK, alpha=0.75,
)
axes[-1].set_xlabel("")
fig.suptitle(
    "Two instruments, two orders of magnitude — separate panels, never twin axes",
    fontsize=13, color=INK, x=0.02, ha="left",
)
fig.tight_layout()
plt.show()


# %% — Helper: correlation heatmap
def corr_heatmap(df, title, ax=None):
    """Lower-triangle correlation heatmap on a diverging scale fixed to [-1, 1].

    Correlation is polarity data: two hues either side of a neutral midpoint,
    never a rainbow. The scale is pinned so colour means the same thing across
    every figure in this notebook — an auto-scaled colour bar would make a
    0.4 look like a 0.95.
    """
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        linewidths=2, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        square=True,
    )
    ax.set_title(title, fontsize=12, loc="left", color=INK, pad=12)
    ax.tick_params(colors=INK, labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return corr


# %% — Correlation on LEVELS (deliberately misleading — read the note)
overlap = panel.loc[wallet_start:, ["n_yape_plin", "v_yape_plin"] + AGGREGATES].dropna()
print(f"Overlap window: {overlap.index.min():%b %Y} → {overlap.index.max():%b %Y}  "
      f"({len(overlap)} months)")

corr_levels = corr_heatmap(
    overlap,
    "Levels — inflated by shared trend, do not act on these",
)
plt.tight_layout()
plt.show()

print(
    "\nExpect almost everything above 0.9. That is what happens when every\n"
    "series is rising: the correlation measures 'both go up', which we already\n"
    "knew. It says nothing about whether they move TOGETHER month to month."
)


# %% — Correlation on GROWTH (the honest measure)
#
# log(x_t) - log(x_{t-1}) ~ monthly growth rate. Differencing removes the
# shared trend, so what survives is co-movement in the deviations. Zeros and
# negatives are excluded because log is undefined there.

growth = np.log(overlap.where(overlap > 0)).diff().dropna()
print(f"Growth observations: {len(growth)}")

corr_growth = corr_heatmap(
    growth,
    "Month-over-month log growth — the correlation to act on",
)
plt.tight_layout()
plt.show()

print(
    f"\nWith n = {len(growth)}, the 95% interval around any r is roughly ±0.35.\n"
    "Treat |r| < 0.4 as indistinguishable from zero, and read the sign and\n"
    "rough magnitude rather than the second decimal place."
)


# %% — Does any aggregate LEAD Yape/Plin?
#
# A feature is only useful for forecasting if it is available before the thing
# being predicted moves. Positive lag = the aggregate leads.

target_growth = growth["n_yape_plin"]
rows = []
for col in AGGREGATES:
    for lag in range(0, 7):
        rows.append({
            "series": col,
            "lag_months": lag,
            "r": target_growth.corr(growth[col].shift(lag)),
        })

lead_lag = pd.DataFrame(rows).pivot(index="lag_months", columns="series", values="r")

print("--- Correlation of n_yape_plin growth with lagged aggregate growth ---")
print(lead_lag.round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 5.5))
for color, col in zip(PALETTE, AGGREGATES, strict=False):
    ax.plot(lead_lag.index, lead_lag[col], marker="o", markersize=7,
            linewidth=2, color=color, label=col)
    ax.annotate(col, xy=(lead_lag.index[-1], lead_lag[col].iloc[-1]),
                xytext=(8, 0), textcoords="offset points",
                fontsize=9, color=INK, va="center")

ax.axhline(0, color=INK, linewidth=1, alpha=0.5)
ax.set_xlabel("Lag applied to the aggregate series (months)", fontsize=10, color=INK)
ax.set_ylabel("Pearson r with n_yape_plin growth", fontsize=10, color=INK)
ax.set_title("Lead–lag structure in monthly growth rates",
             fontsize=12, loc="left", color=INK, pad=10)
ax.set_ylim(-1, 1)
ax.grid(True, axis="y", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=INK, labelsize=9)
ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.tight_layout()
plt.show()


# %% [markdown]
# ## How to read the result
#
# **A high growth correlation at lag 0** means the aggregate moves with the
# wallets contemporaneously. Useful for explanation, useless for forecasting —
# you would need next month's value of the feature to predict next month's
# target.
#
# **A high correlation at lag ≥ 1** is the interesting case: the aggregate
# leads, and can be used as a genuine predictor.
#
# **A near-zero correlation on growth despite ~0.99 on levels** means the two
# series share nothing but a trend. Still potentially useful for the Phase 2
# panel model — which learns from the *shape* of long adoption curves — but not
# as a Phase 1 feature.
#
# Expect `n_transf_intra_agg` to be the strongest: Yape and Plin are literally
# a component of it, so contemporaneous correlation is close to mechanical.
# That makes it excellent for the panel model's pre-2024 history and weak as an
# independent predictor. `n_dinero_electronico` is the genuine test — a separate
# instrument, so any co-movement reflects shared adoption dynamics rather than
# arithmetic.
