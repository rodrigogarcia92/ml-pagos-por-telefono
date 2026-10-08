"""Google Trends data gate G1-G3 (plan 9.2, O-14). Read-only.

    .venv\\Scripts\\python.exe scripts/trends_gate.py
    .venv\\Scripts\\python.exe scripts/trends_gate.py --snapshot data/processed/panel_20261005T000000Z.parquet

Not part of training: it imports no model code, writes no MLflow run and no
warehouse row, and its only output is stdout plus reports/trends_gate_{date}.md.
Works with ONE pull (G1 and G3 run, G2 is reported "pending").

THE CV PERIOD, AND ONLY THE CV PERIOD
-------------------------------------
Every number below is computed on months BEFORE the final holdout. The holdout is
the last HOLDOUT_MONTHS (12) target months of the payments series on the snapshot
(plan 6.1), so the gate reads the snapshot only to find where the
payments end and to get n_transf_intra_agg for G3; every month at or after the
holdout start is dropped before any statistic is taken. On snapshot 20261005
(payments through 2026-07) the CV period ends 2025-07.

THE THRESHOLDS ARE THE PLAN'S AND ARE NOT OPTIONS
-------------------------------------------------
G1  months from 2017-09 with the consolidated index < 5 .......... pass if <= 6
                                           fail -> run the Trends test on w2021
G2  corr(dlog gt) between two pulls .............................. pass if >= 0.90
                                           fail -> third pull + geometric mean
G3  max over leads 0..3 of corr(dlog gt_m, dlog n_{m+lead}) ...... pass if >= 0.20
                                           (positive)  fail -> negative result,
                                           s5 runs FS3 only (12 parents)

Log differences are undefined where the index is 0 (the plan's "<1 -> 0.5" rule
assumes the index never prints a bare 0). The gate drops those months pairwise and
REPORTS how many pairs each statistic rests on, and when a full 13-month history for
gt_ma12 first exists. It does not repair, floor or smooth anything.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_collection import trends  # noqa: E402

HOLDOUT_MONTHS = 12                       # train.HOLDOUT_MONTHS["w2019"], restated: no model import
W2019_FIRST_TARGET = pd.Timestamp("2019-01-01")
G1_FROM = pd.Timestamp("2017-09-01")
G1_FLOOR = 5.0
G1_MAX_LOW_MONTHS = 6
G2_MIN_CORR = 0.90
G3_MIN_CORR = 0.20
G3_LEADS = (0, 1, 2, 3)
TARGET_COL = "n_transf_intra_agg"
REPORT_DIR = Path("reports")


@dataclass
class Gate:
    id: str
    name: str
    status: str                  # PASS | FAIL | PENDING
    rule: str
    result: str
    consequence: str = ""
    detail: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# CV period
# --------------------------------------------------------------------------- #
def cv_end_from_payments(payments: pd.Series, holdout: int = HOLDOUT_MONTHS) -> pd.Timestamp:
    """Last CV target month = the month before the final `holdout` published months."""
    last = payments.dropna().index.max()
    return last - pd.DateOffset(months=holdout)


def dlog(s: pd.Series) -> pd.Series:
    """Log difference; NaN wherever either level is not strictly positive."""
    s = s.astype(float)
    pos = s.where(s > 0)
    return np.log(pos).diff()


def first_full_history(s: pd.Series, n_levels: int = 13) -> pd.Timestamp | None:
    """First month m such that s is > 0 on m-12 .. m (what gt_ma12 needs at origin m)."""
    ok = (s > 0).astype(int).rolling(n_levels).sum() == n_levels
    return ok.index[ok.to_numpy()].min() if ok.any() else None


# --------------------------------------------------------------------------- #
# G1, G2, G3
# --------------------------------------------------------------------------- #
def g1_coverage(gt: pd.Series, cv_end: pd.Timestamp) -> Gate:
    s = gt[(gt.index >= G1_FROM) & (gt.index <= cv_end)]
    gaps = len(pd.date_range(s.index.min(), s.index.max(), freq="MS")) - len(s)
    low = s[s < G1_FLOOR]
    ok = len(low) <= G1_MAX_LOW_MONTHS and gaps == 0
    detail = [f"window {s.index.min():%Y-%m} .. {s.index.max():%Y-%m}: {len(s)} months, {gaps} gap(s)",
              f"months below {G1_FLOOR:g}: {len(low)}"
              + (f" (first {low.index.min():%Y-%m}, last {low.index.max():%Y-%m})" if len(low) else ""),
              f"months exactly 0: {int((s == 0).sum())}"]
    return Gate(
        "G1", "Coverage", "PASS" if ok else "FAIL",
        f"<= {G1_MAX_LOW_MONTHS} months from {G1_FROM:%Y-%m} with index < {G1_FLOOR:g}, gap-free",
        f"{len(low)} month(s) below {G1_FLOOR:g}",
        "" if ok else "run the Trends test on w2021 (folds 29 / 27 / 26) and say so (plan 9.2)",
        detail,
    )


def g2_stability(pulls: list[trends.Pull], cv_end: pd.Timestamp) -> Gate:
    rule = f"corr of dlog gt between two pulls >= {G2_MIN_CORR:.2f}"
    if len(pulls) < 2:
        return Gate("G2", "Stability", "PENDING", rule, f"{len(pulls)} pull(s); a second pull on a "
                    "different day is needed", "pending — nothing is concluded from one pull")
    series = {p.pull_date: trends.consolidated(p.data) for p in pulls}
    rows, worst = [], 1.0
    names = sorted(series)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x = dlog(series[a][series[a].index <= cv_end])
            y = dlog(series[b][series[b].index <= cv_end])
            j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
            r = float(j["a"].corr(j["b"])) if len(j) > 2 else float("nan")
            rows.append(f"{a} vs {b}: corr {r:+.3f} on {len(j)} month-pairs")
            worst = min(worst, r) if np.isfinite(r) else float("nan")
    ok = bool(np.isfinite(worst) and worst >= G2_MIN_CORR)
    return Gate("G2", "Stability", "PASS" if ok else "FAIL", rule, f"min corr {worst:+.3f}",
                "" if ok else "take a third pull and use the geometric mean of the pulls, "
                              "declared in the .meta.json (plan 9.2)", rows)


def g3_relevance(gt: pd.Series, payments: pd.Series, cv_end: pd.Timestamp) -> Gate:
    rule = (f"max corr(dlog gt_m, dlog n_(m+lead)), lead in {list(G3_LEADS)}, "
            f">= {G3_MIN_CORR:.2f} and positive")
    n_dlog = dlog(payments[(payments.index <= cv_end)])
    n_dlog = n_dlog[n_dlog.index >= W2019_FIRST_TARGET]            # CV target months of w2019
    g_dlog = dlog(gt[gt.index <= cv_end])
    rows, corrs = [], {}
    for lead in G3_LEADS:
        # gt month m pairs with the payments month m + lead
        g = g_dlog.copy()
        g.index = g.index + pd.DateOffset(months=lead)
        j = pd.concat([g, n_dlog], axis=1, keys=["gt", "n"]).dropna()
        j = j[j.index <= cv_end]
        r = float(j["gt"].corr(j["n"])) if len(j) > 2 and j["gt"].std() > 0 else float("nan")
        corrs[lead] = r
        rows.append(f"lead {lead}: corr {r:+.3f} on {len(j)} month-pairs"
                    + (f" ({j.index.min():%Y-%m} .. {j.index.max():%Y-%m})" if len(j) else ""))
    finite = {k: v for k, v in corrs.items() if np.isfinite(v)}
    if not finite:
        return Gate("G3", "Relevance", "FAIL", rule, "no lead has enough month-pairs",
                    "Trends is a negative result; s5 runs FS3 only (12 parents) (plan 9.2)", rows)
    best = max(finite, key=finite.get)
    ok = finite[best] >= G3_MIN_CORR
    return Gate("G3", "Relevance", "PASS" if ok else "FAIL", rule,
                f"max {finite[best]:+.3f} at lead {best}",
                "" if ok else "Trends is a negative result; s5 runs FS3 only (12 parents) (plan 9.2)",
                rows)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(pulls: list[trends.Pull], payments: pd.Series, *, use_pull: str | None = None):
    cv_end = cv_end_from_payments(payments)
    chosen = next((p for p in pulls if p.pull_date == use_pull), pulls[-1]) if pulls else None
    gt_full = trends.consolidated(chosen.data)
    gt = gt_full[gt_full.index <= cv_end]                   # NEVER past the CV period
    gates = [g1_coverage(gt, cv_end), g2_stability(pulls, cv_end), g3_relevance(gt, payments, cv_end)]
    notes = diagnostics(gt, cv_end, chosen)
    return gates, cv_end, chosen, notes


def diagnostics(gt: pd.Series, cv_end: pd.Timestamp, chosen: trends.Pull) -> list[str]:
    zero = gt[gt <= 0]
    first = first_full_history(gt)
    notes = [
        f"pull used for G1/G3: {chosen.pull_date}  (CV months {gt.index.min():%Y-%m} .. {gt.index.max():%Y-%m})",
        f"consolidated = yape + plin; yape mean {chosen.data.loc[chosen.data['date'] <= cv_end, 'yape'].mean():.1f}, "
        f"plin mean {chosen.data.loc[chosen.data['date'] <= cv_end, 'plin'].mean():.1f} (Yape-dominated; plan 4.5)",
        f"months with index 0 (log undefined): {len(zero)}"
        + (f", last {zero.index.max():%Y-%m}" if len(zero) else ""),
        "first month with 13 consecutive positive levels (gt_ma12 computable at that origin): "
        + (f"{first:%Y-%m}" if first is not None else "none in the CV period"),
        "Trends 'Nota' annotation (~2022) falls inside the CV period: "
        + ("YES" if ((gt.index.year == 2022).any()) else "no") + " (information only; no adjustment)",
    ]
    return notes


def render(gates: list[Gate], cv_end: pd.Timestamp, chosen, notes: list[str], today: date) -> str:
    out = [f"# Google Trends data gate — {today:%Y-%m-%d}", "",
           f"Computed on the CV period only (months <= {cv_end:%Y-%m}); the final {HOLDOUT_MONTHS} "
           "target months are never read. Thresholds fixed in training plan §9.2 (v1.7).", "",
           "| Gate | Status | Rule | Result |", "|---|---|---|---|"]
    out += [f"| {g.id} {g.name} | **{g.status}** | {g.rule} | {g.result} |" for g in gates]
    out += [""]
    for g in gates:
        out += [f"## {g.id} — {g.name}: {g.status}", ""] + [f"- {d}" for d in g.detail]
        if g.consequence:
            out += ["", f"**Consequence:** {g.consequence}"]
        out += [""]
    out += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--snapshot", default=None,
                    help="panel_*.parquet to take n_transf_intra_agg from (default: newest)")
    ap.add_argument("--dir", default=str(trends.RAW_DIR))
    ap.add_argument("--pull", default=None, help="pull_date used for G1/G3 (default: latest)")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()

    pulls = trends.load_pulls(Path(a.dir))
    if not pulls:
        raise SystemExit(f"no *_yape_plin.csv in {a.dir}")
    snap = Path(a.snapshot) if a.snapshot else sorted(Path("data/processed").glob("panel_*.parquet"))[-1]
    panel = pd.read_parquet(snap)
    panel.index = pd.to_datetime(panel.index)
    payments = panel[TARGET_COL].sort_index()

    gates, cv_end, chosen, notes = run(pulls, payments, use_pull=a.pull)
    print(f"snapshot : {snap.name}   pulls: {', '.join(p.pull_date for p in pulls)}")
    print(f"CV period: target months <= {cv_end:%Y-%m}  (holdout = the final {HOLDOUT_MONTHS}, not read)\n")
    for g in gates:
        print(f"{g.id} {g.name:<10} {g.status:<8} {g.result}")
        print(f"   rule: {g.rule}")
        for d in g.detail:
            print(f"   - {d}")
        if g.consequence:
            print(f"   => {g.consequence}")
    print()
    for n in notes:
        print(f"* {n}")

    if not a.no_report:
        REPORT_DIR.mkdir(exist_ok=True)
        path = REPORT_DIR / f"trends_gate_{date.today():%Y-%m-%d}.md"
        path.write_text(render(gates, cv_end, chosen, notes, date.today()), encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
