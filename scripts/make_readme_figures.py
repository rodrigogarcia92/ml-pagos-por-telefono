"""Regenerate the three README figures into docs/figures/. Read-only on data and MLflow.

    .venv\\Scripts\\python.exe scripts/make_readme_figures.py
    .venv\\Scripts\\python.exe scripts/make_readme_figures.py --snapshot data/processed/panel_20261005T000000Z.parquet

What it does (target t3, horizon 3, window w2019, cross-validation only -- the holdout
is never touched):

  fig1_forecasts.png  Re-fits SVR, random forest and XGBoost on FS3 with the hyperparameters
                      logged in MLflow for the protocol-1.7 `ens3` run, on the SAME expanding
                      folds the sweep used, and plots the ensemble's 3-month-ahead forecasts
                      and the naive_drift trend line against the actual series.
  fig2_selection.png  Reads the per-fold MASE that MLflow logged for each candidate and plots
                      the paired difference to the leader +- 1 standard error.
  fig3_drivers.png    Grouped permutation importance of the ensemble, out-of-sample: for each
                      fold, one feature block in the test row is replaced by that block from
                      random training rows, and the increase in absolute error is recorded.

Nothing here is a protocol run: no MLflow run is created, nothing is tuned, no model is
selected. Same inputs give the same figures (fixed seeds), up to small XGBoost float noise.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_training import dataset, registry, splits, train  # noqa: E402

TARGET, HORIZON, WINDOW, FEATURE_SET = "t3", 3, "w2019", "FS3_activity"
MEMBERS = ("svr_rbf", "rf", "xgboost")
PERM_DRAWS = 40
SEED = 0

# Palette (light surface). Blue = the model, orange = the trend baseline, ink = actual.
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Candidates shown in the selection chart: (model_family, feature_set) -> plain-English label.
CANDIDATES = {
    ("ens3", "FS3_activity"): "Ensemble · production set",
    ("xgboost", "FS5_macro_full"): "XGBoost · all macro variables",
    ("xgboost", "FS5b_stance"): "XGBoost · macro + policy stance",
    ("svr_rbf", "FS3_activity"): "SVR · production set",
    ("xgboost", "FS3_activity"): "XGBoost · production set",
    ("rf", "FS3_activity"): "Random Forest · production set",
    ("svr_rbf", "FS4_prices"): "SVR · production set + prices",
    ("ridge", "FS3_activity"): "Ridge (linear) · production set",
    ("naive_drift", "none"): "Simple trend line",
}
LEADER = ("ens3", "FS3_activity")
BASELINE = ("naive_drift", "none")


# --------------------------------------------------------------------------- #
# MLflow (read-only, straight from the SQLite backend)
# --------------------------------------------------------------------------- #
def _cast(v: str):
    if v == "None":
        return None
    if v in ("scale", "auto"):
        return v
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v else f
    except ValueError:
        return v


def _parents(con: sqlite3.Connection, data_version: str) -> pd.DataFrame:
    keys = ("model_family", "feature_set", "target_id", "window", "stage", "data_version", "protocol_version")
    t = pd.read_sql(
        f"select run_uuid, key, value from tags where key in {keys}", con
    ).pivot(index="run_uuid", columns="key", values="value")
    runs = pd.read_sql("select run_uuid, status, start_time from runs", con).set_index("run_uuid")
    t = t.join(runs)
    t = t[(t.target_id == TARGET) & (t.window == WINDOW) & (t.stage == "cv")
          & (t.data_version == data_version) & (t.status == "FINISHED")]
    # One run per (family, feature set): the newest protocol, then the newest start time.
    return t.sort_values(["protocol_version", "start_time"]).groupby(["model_family", "feature_set"]).tail(1)


def member_params(con: sqlite3.Connection, parents: pd.DataFrame) -> dict[str, dict]:
    ens = parents[(parents.model_family == "ens3") & (parents.feature_set == FEATURE_SET)]
    if ens.empty:
        raise SystemExit("No finished ens3 / FS3 run for t3 w2019 on this data_version in MLflow.")
    rows = con.execute("select key, value from params where run_uuid = ?", (ens.index[0],)).fetchall()
    out: dict[str, dict] = {m: {} for m in MEMBERS}
    for k, v in rows:
        if k.startswith("member_") and "__" in k:
            fam, p = k.removeprefix("member_").split("__", 1)
            if fam in out:
                out[fam][p] = _cast(v)
    missing = [m for m, p in out.items() if not p]
    if missing:
        raise SystemExit(f"ens3 run has no logged hyperparameters for {missing}.")
    return out


def fold_mase(con: sqlite3.Connection, parents: pd.DataFrame) -> pd.DataFrame:
    want = parents[[ (f, s) in CANDIDATES for f, s in zip(parents.model_family, parents.feature_set)]]
    ids = tuple(want.index) + ("",)
    ch = pd.read_sql(
        "select p.run_uuid child, p.value parent, n.value name from tags p "
        "join tags n on n.run_uuid = p.run_uuid and n.key = 'mlflow.runName' "
        f"where p.key = 'mlflow.parentRunId' and p.value in {ids}", con)
    m = pd.read_sql("select run_uuid, value from metrics where key = 'mase'", con)
    m = m.merge(ch, left_on="run_uuid", right_on="child")
    m["fold"] = m["name"].str.extract(r"(\d+)$").astype(int)
    wide = m.pivot_table(index="fold", columns="parent", values="value")
    wide.columns = [CANDIDATES[(want.loc[c, "model_family"], want.loc[c, "feature_set"])] for c in wide.columns]
    return wide


# --------------------------------------------------------------------------- #
# Re-fit the ensemble on the sweep's folds
# --------------------------------------------------------------------------- #
def build_frames(panel, meta):
    fs = train.load_feature_sets()[FEATURE_SET]
    return {
        fam: dataset.build(panel, meta, target_id=TARGET, horizon=HORIZON, window=WINDOW,
                           columns=registry.model_columns(fam, fs),
                           encoding=registry.default_encoding(fam))
        for fam in MEMBERS
    }


def _block(col: str) -> str | None:
    if col.startswith("d_covid"):
        return None          # a 2020 pulse; permuting it plants COVID into 2022+ rows (artefact)
    if col.startswith("cal"):
        return "calendar (days, weekends, holidays, month)"
    return {"y": "past transfers (own history)", "circ": "cash in circulation",
            "pbi": "economic activity (GDP index)"}.get(col.split("_")[0], col)


def refit(frames: dict, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold-by-fold forecasts (levels) and grouped permutation importance of the ensemble."""
    rng = np.random.default_rng(SEED)
    ref = frames["rf"]
    n_cv = len(ref.X) - train.HOLDOUT_MONTHS[WINDOW]
    folds = splits.make_folds(n_cv, window=WINDOW, horizon=HORIZON)
    kappa_shift = pd.DateOffset(months=HORIZON - 2)   # target month = origin + (h - kappa), kappa = 2

    blocks: dict[str, dict[str, list[int]]] = {}
    for fam, fr in frames.items():
        for j, c in enumerate(fr.X.columns):
            b = _block(c)
            if b:
                blocks.setdefault(b, {}).setdefault(fam, []).append(j)

    rows, extra = [], {b: [] for b in blocks}
    for f in folds:
        tr, te_idx = f.train, f.test
        te = te_idx[0]
        anchor, actual = ref.ctx["anchor"].iloc[te], ref.ctx["y_level"].iloc[te]
        mats, models, z = {}, {}, {}
        for fam, fr in frames.items():
            Xa = fr.X.to_numpy()
            Xtr, Xte, _ = train._prepare(Xa[tr], Xa[te_idx], Xa[tr[-1] + 1: te])
            mdl = registry.build(fam, params[fam], horizon=HORIZON)
            mdl.fit(Xtr, fr.y.to_numpy()[tr], None)
            mats[fam], models[fam], z[fam] = (Xtr, Xte), mdl, mdl.predict(Xte, None)[0]
        z_ens = float(np.mean(list(z.values())))
        # naive_drift: average h-step log growth over the training rows (registry.NaiveDrift)
        d = np.log(ref.ctx["y_level"].to_numpy()[tr]) - np.log(ref.ctx["anchor"].to_numpy()[tr])
        rows.append({"target": ref.X.index[te] + kappa_shift, "actual": actual,
                     "ensemble": anchor * np.exp(z_ens),
                     "trend": anchor * np.exp(np.nanmean(d)),
                     **{fam: anchor * np.exp(v) for fam, v in z.items()}})

        e0 = abs(anchor * np.exp(z_ens) - actual)
        for b, cols in blocks.items():
            draws = rng.integers(0, len(tr), PERM_DRAWS)
            preds = []
            for fam, (Xtr, Xte) in mats.items():
                Xp = np.repeat(Xte, PERM_DRAWS, axis=0)
                if fam in cols:
                    Xp[:, cols[fam]] = Xtr[draws][:, cols[fam]]
                preds.append(models[fam].predict(Xp, None))
            ep = np.abs(anchor * np.exp(np.mean(preds, axis=0)) - actual)
            extra[b].append(ep.mean() - e0)

    fc = pd.DataFrame(rows).set_index("target")
    mae = float(np.abs(fc.ensemble - fc.actual).mean())
    imp = pd.DataFrame([
        {"block": b, "pct": 100 * np.mean(v) / mae,
         "se": 100 * np.std(v, ddof=1) / np.sqrt(len(v)) / mae}
        for b, v in extra.items()
    ]).sort_values("pct")
    return fc, imp


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })


def fig_forecasts(fc: pd.DataFrame, out: Path):
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7.2), gridspec_kw={"height_ratios": [3, 2]}, sharex=True)
    a.plot(fc.index, fc.actual, color=INK, lw=2, label="Actual")
    a.plot(fc.index, fc.ensemble, color=BLUE, lw=1.8, label="Ensemble (SVR + RF + XGBoost)")
    a.plot(fc.index, fc.trend, color=ORANGE, lw=1.8, ls="--", label="Simple trend line")
    a.set_ylabel("Transfers per month (millions)")
    a.legend(frameon=False, loc="upper left")
    a.set_title("Each point is a 3-month-ahead forecast, made with only the data published at the time",
                loc="left", fontsize=11, color=INK)
    e_m = (fc.ensemble - fc.actual) / fc.actual * 100
    e_t = (fc.trend - fc.actual) / fc.actual * 100
    off = pd.Timedelta(days=4.5)
    b.bar(fc.index - off, e_m, width=9, color=BLUE, label="Ensemble")
    b.bar(fc.index + off, e_t, width=9, color=ORANGE, label="Simple trend")
    b.axhline(0, color=MUTED, lw=.8)
    b.set_ylabel("Forecast error (%)\n(+ over, − under)")
    b.set_ylim(-22, 20)
    b.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_selection(M: pd.DataFrame, out: Path):
    lead = CANDIDATES[LEADER]
    r = pd.DataFrame([
        {"label": c, "mase": M[c].mean(), "d": (M[c] - M[lead]).mean(),
         "se": (M[c] - M[lead]).std(ddof=1) / np.sqrt(len(M))}
        for c in M.columns
    ]).sort_values("mase", ascending=False)
    fig, a = plt.subplots(figsize=(10, 5.2))
    for i, x in enumerate(r.itertuples()):
        col = ORANGE if x.label == CANDIDATES[BASELINE] else (BLUE if x.label == lead else MUTED)
        a.errorbar(x.d, i, xerr=x.se, fmt="o", color=col, ms=7, capsize=3, lw=1.4)
        a.text(0.215, i, f"MASE {x.mase:.3f}", va="center", fontsize=9, color=MUTED)
    a.set_yticks(range(len(r)))
    a.set_yticklabels(r.label, color=INK)
    a.axvline(0, color=MUTED, lw=.8)
    a.set_xlim(-0.04, 0.27)
    a.set_xlabel("Extra error vs the leader (paired, MASE) ± 1 standard error")
    a.set_title("Who is statistically tied with the leader? Those whose bar touches zero",
                loc="left", fontsize=11, color=INK)
    a.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_drivers(imp: pd.DataFrame, out: Path):
    fig, a = plt.subplots(figsize=(10, 3.8))
    a.barh(imp.block, imp.pct, xerr=imp.se, color=BLUE, height=.5,
           error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
    for y, (v, s) in enumerate(zip(imp.pct, imp.se)):
        a.text(v + s + 2, y, f"+{v:.0f}%", va="center", fontsize=9, color=INK)
    a.set_xlabel("Extra forecast error if this block is scrambled (% of the ensemble's error)")
    a.set_xlim(0, max(100, float((imp.pct + imp.se).max()) + 15))
    a.grid(axis="y", visible=False)
    a.set_title("What the ensemble relies on (out-of-sample)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--snapshot", default=None, help="panel_*.parquet (default: newest in data/processed)")
    ap.add_argument("--db", default=str(ROOT / "mlflow" / "mlflow.db"), help="MLflow SQLite backend")
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    a = ap.parse_args()

    snap = Path(a.snapshot) if a.snapshot else sorted((ROOT / "data" / "processed").glob("panel_*.parquet"))[-1]
    data_version = snap.stem.removeprefix("panel_")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"snapshot {snap.name} | MLflow {a.db} | output {out}")

    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)   # read-only
    parents = _parents(con, data_version)
    params = member_params(con, parents)
    M = fold_mase(con, parents)
    con.close()

    panel, meta = dataset.load_snapshot(snap)
    print("re-fitting the ensemble on the sweep's folds (about 1-3 minutes)...")
    fc, imp = refit(build_frames(panel, meta), params)

    _style()
    fig_forecasts(fc, out / "fig1_forecasts.png")
    fig_selection(M, out / "fig2_selection.png")
    fig_drivers(imp, out / "fig3_drivers.png")

    mape = lambda c: 100 * np.mean(np.abs(fc[c] - fc.actual) / fc.actual)  # noqa: E731
    print(f"{len(fc)} forecasts, {fc.index.min():%Y-%m} .. {fc.index.max():%Y-%m}")
    print(f"MAPE  ensemble {mape('ensemble'):.1f}%   trend {mape('trend'):.1f}%")
    print(imp.sort_values('pct', ascending=False).round(1).to_string(index=False))
    print(f"wrote {out / 'fig1_forecasts.png'}, fig2_selection.png, fig3_drivers.png")


if __name__ == "__main__":
    main()
