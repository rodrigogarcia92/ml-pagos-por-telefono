"""Google Trends ingestion (training_plan.md 4.5, 9.2): parser, sidecar, loader.

Offline: every test writes its own CSVs into tmp_path. The real pull in
data/raw/google_trends/ is read once, read-only, as a regression check that the
shipped file still parses.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data_collection import load_trends, trends
from src.data_collection.trends import TrendsFormatError

REAL = Path("data/raw/google_trends/2026-10-05_yape_plin.csv")


def _months(n, start="2017-01-01"):
    return pd.date_range(start, periods=n, freq="MS")


def _csv(path, rows, header='"Time","Yape","Plin"', preamble=(), delim=","):
    lines = list(preamble) + [header.replace(",", delim)]
    lines += [delim.join(str(c) for c in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rows(n=24, fmt="%Y-%m-%d"):
    return [(d.strftime(fmt), i % 50, (i % 7)) for i, d in enumerate(_months(n))]


def _pull(tmp_path, pull="2019-01-10", n=24, **kw):
    """A valid CSV + sidecar whose last complete month is the month before `pull`."""
    csv = _csv(tmp_path / f"{pull}_yape_plin.csv", _rows(n), **kw)
    meta = trends.build_meta(csv)
    trends.meta_path(csv).write_text(json.dumps(meta), encoding="utf-8")
    return csv


# --------------------------------------------------------------------------- #
# parse_trends_csv
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL.exists(), reason="no local Trends pull")
def test_the_shipped_pull_parses_and_matches_the_plan_description():
    df = trends.parse_trends_csv(REAL)
    assert list(df.columns) == ["date", "yape", "plin"]
    assert df["date"].iloc[0] == pd.Timestamp("2017-01-01")
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
    # Plan 4.5 (v1.7.1): Plin ~0 until mid-2023, so the sum is Yape-dominated.
    assert df["plin"][df["date"] < "2023-07-01"].max() == 0
    assert df["yape"].mean() > 5 * df["plin"].mean()


def test_the_header_is_detected_not_hardcoded_spanish_locale_with_preamble(tmp_path):
    rows = _rows(24, fmt="%Y-%m")
    rows[3] = ("2017-04", "<1", "<1")
    p = _csv(
        tmp_path / "x.csv", rows,
        header='Mes;Yape (Aplicación): (Perú);Plin (Tema): (Perú)',
        preamble=["Categoría: Todas las categorías", "", ],
        delim=";",
    )
    df = trends.parse_trends_csv(p)
    assert len(df) == 24
    assert df.loc[3, ["yape", "plin"]].tolist() == [0.5, 0.5]       # "<1" -> 0.5 (plan 4.5)


def test_column_order_in_the_file_does_not_matter(tmp_path):
    rows = [(d, b, a) for d, a, b in _rows(24)]                      # plin first, yape second
    p = _csv(tmp_path / "x.csv", rows, header='"Time","Plin","Yape"')
    df = trends.parse_trends_csv(p)
    assert df["yape"].tolist() == [r[2] for r in rows]


def test_a_literal_zero_stays_zero_it_is_not_the_subone_marker(tmp_path):
    rows = _rows(24)
    rows[0] = ("2017-01-01", 0, 0)
    assert trends.parse_trends_csv(_csv(tmp_path / "x.csv", rows)).loc[0, "yape"] == 0.0


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r[:5] + r[6:], "not contiguous"),
    (lambda r: r + [r[-1]], "duplicate"),
    (lambda r: r[1:], "first month"),
    (lambda r: r[:2] + [(r[2][0], 101, 0)] + r[3:], "outside 0..100"),
    (lambda r: r[:2] + [(r[2][0], "abc", 0)] + r[3:], "neither a number"),
    (lambda r: r[:2] + [("2017-03-15", 1, 1)] + r[3:], "not monthly"),
])
def test_validation_fails_loudly_and_names_the_line(tmp_path, mutate, match):
    p = _csv(tmp_path / "x.csv", mutate(_rows(24)))
    with pytest.raises(TrendsFormatError, match=match) as e:
        trends.parse_trends_csv(p)
    assert "x.csv" in str(e.value)


def test_an_unexpected_format_raises_with_the_offending_line(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text('"Time","Yape"\n"2017-01-01",1\n', encoding="utf-8")
    with pytest.raises(TrendsFormatError, match="two entity columns") as e:
        trends.parse_trends_csv(p)
    assert '"Time","Yape"' in str(e.value)

    p.write_text('"Time","Foo","Bar"\n"2017-01-01",1,2\n', encoding="utf-8")
    with pytest.raises(TrendsFormatError, match="mentioning 'yape'"):
        trends.parse_trends_csv(p)

    p.write_text("nothing\nhere\n", encoding="utf-8")
    with pytest.raises(TrendsFormatError, match="no data line"):
        trends.parse_trends_csv(p)

    _csv(p, _rows(24))
    p.write_text(p.read_text() + "Nota: cambio de sistema\n", encoding="utf-8")
    with pytest.raises(TrendsFormatError, match="not a data line"):
        trends.parse_trends_csv(p)


def test_consolidated_is_the_sum_in_index_points(tmp_path):
    df = trends.parse_trends_csv(_csv(tmp_path / "x.csv", _rows(24)))
    s = trends.consolidated(df)
    assert s.tolist() == (df["yape"] + df["plin"]).tolist()
    assert s.index.equals(pd.DatetimeIndex(df["date"]))


# --------------------------------------------------------------------------- #
# complete months, sidecar, selections
# --------------------------------------------------------------------------- #
def test_the_incomplete_pull_month_is_dropped_and_a_stale_pull_is_refused(tmp_path):
    # 24 rows = 2017-01 .. 2018-12 ; a pull on 2018-12-20 has 2018-12 incomplete.
    df = trends.parse_trends_csv(_csv(tmp_path / "x.csv", _rows(24)))
    out = trends.complete_months(df, "2018-12-20")
    assert out["date"].iloc[-1] == pd.Timestamp("2018-11-01") and len(out) == 23
    with pytest.raises(TrendsFormatError, match="last complete month"):
        trends.complete_months(df, "2019-03-01")                      # file stops short


def test_a_csv_without_a_sidecar_is_refused(tmp_path):
    csv = _csv(tmp_path / "2019-01-10_yape_plin.csv", _rows(24))
    with pytest.raises(TrendsFormatError, match="no sidecar"):
        trends.load_pull(csv)


def test_sidecar_records_the_selections_and_the_request(tmp_path):
    pull = trends.load_pull(_pull(tmp_path))
    m = pull.meta
    assert m["selections"] == [
        {"column": "yape", "name": "Yape", "type": "Aplicación"},
        {"column": "plin", "name": "Plin", "type": "Tema"},
    ]
    assert (m["geo"], m["category"], m["property"], m["request_count"]) == (
        "PE", "All categories", "Web search", 1)
    assert m["period_start"] == "2017-01-01" and m["pull_date"] == "2019-01-10"
    assert pull.data["date"].iloc[-1] == pd.Timestamp("2018-12-01")


def test_two_requests_are_refused_they_would_not_share_a_scale(tmp_path):
    csv = _pull(tmp_path)
    mp = trends.meta_path(csv)
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["request_count"] = 2
    mp.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(TrendsFormatError, match="ONE"):
        trends.load_pull(csv)


def test_a_pull_whose_selections_differ_from_the_plan_is_refused(tmp_path):
    csv = _pull(tmp_path)
    mp = trends.meta_path(csv)
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["selections"][0]["type"] = "Término de búsqueda"               # plain term, not the entity
    mp.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(TrendsFormatError, match="selections"):
        trends.load_pull(csv)


def test_pulls_with_mismatching_selections_are_refused_by_the_cross_check(tmp_path):
    a = trends.load_pull(_pull(tmp_path, "2019-01-10"))
    b = trends.load_pull(_pull(tmp_path, "2019-01-11"))
    trends.check_same_selections([a, b])                              # identical: fine
    b2 = trends.Pull(b.pull_date, b.csv, {**b.meta, "geo": "CL"}, b.raw, b.data)
    with pytest.raises(TrendsFormatError, match="differs from pull"):
        trends.check_same_selections([a, b2])
    b3 = trends.Pull(b.pull_date, b.csv,
                     {**b.meta, "selections": b.meta["selections"][::-1]}, b.raw, b.data)
    with pytest.raises(TrendsFormatError, match="differs from pull"):
        trends.check_same_selections([a, b3])


def test_a_csv_edited_after_its_sidecar_was_written_is_refused(tmp_path):
    csv = _pull(tmp_path)
    csv.write_text(csv.read_text(encoding="utf-8").replace('2017-02-01,1,1', '2017-02-01,2,1'),
                   encoding="utf-8")
    with pytest.raises(TrendsFormatError, match="sha256"):
        trends.load_pull(csv)


def test_sidecar_hash_ignores_line_endings(tmp_path):
    csv = _pull(tmp_path)
    # write_text on Windows already wrote CRLF; normalise first so this really flips LF <-> CRLF.
    lf = csv.read_bytes().replace(b"\r\n", b"\n")
    csv.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert b"\r\n" in csv.read_bytes()
    csv.write_bytes(lf)
    trends.load_pull(csv)
    csv.write_bytes(lf.replace(b"\n", b"\r\n"))
    trends.load_pull(csv)                                             # git on Windows may do this


# --------------------------------------------------------------------------- #
# final-month rule (plan 4.5, v1.7.1)
# --------------------------------------------------------------------------- #
def _two_pulls(tmp_path, delta):
    a = _pull(tmp_path, "2019-01-10")
    rows = _rows(24)
    last = rows[-2]                                                    # 2018-12 is index -1; -2 = 2018-11
    rows[-2] = (last[0], last[1] + delta, last[2])
    csv_b = _csv(tmp_path / "2019-01-11_yape_plin.csv", rows)
    trends.meta_path(csv_b).write_text(json.dumps(trends.build_meta(csv_b)), encoding="utf-8")
    return a, csv_b


def test_final_month_kept_when_pulls_agree_within_ten_points(tmp_path):
    # Both pulls were taken in 2019-01, so the last complete month is 2018-12 (index -1).
    a, b = trends.load_pull(_pull(tmp_path, "2019-01-10")), None
    rows = _rows(24)
    rows[-1] = (rows[-1][0], rows[-1][1] + 10, rows[-1][2])           # exactly 10: not "more than 10"
    csv_b = _csv(tmp_path / "2019-01-11_yape_plin.csv", rows)
    trends.meta_path(csv_b).write_text(json.dumps(trends.build_meta(csv_b)), encoding="utf-8")
    b = trends.load_pull(csv_b)
    res = trends.final_month_check(a, b)
    assert res["month"] == "2018-12" and res["abs_difference"] == 10
    assert res["last_month_dropped"] is False


def test_final_month_dropped_when_pulls_differ_by_more_than_ten_and_recorded(tmp_path):
    _pull(tmp_path, "2019-01-10")
    rows = _rows(24)
    rows[-1] = (rows[-1][0], rows[-1][1] + 11, rows[-1][2])
    csv_b = _csv(tmp_path / "2019-01-11_yape_plin.csv", rows)
    trends.meta_path(csv_b).write_text(json.dumps(trends.build_meta(csv_b)), encoding="utf-8")

    res = trends.reconcile(tmp_path)
    assert res["last_month_dropped"] is True and res["month"] == "2018-12"
    # ...recorded in BOTH sidecars, and honoured on the next load.
    for p in trends.load_pulls(tmp_path):
        assert p.meta["final_month_check"]["last_month_dropped"] is True
        assert p.data["date"].iloc[-1] == pd.Timestamp("2018-11-01")
        assert len(p.data) == 23


def test_reconcile_with_one_pull_is_pending(tmp_path):
    _pull(tmp_path)
    assert trends.reconcile(tmp_path)["status"] == "pending"


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
def test_loader_long_format_is_complete_months_only_one_pull_id_per_pull(tmp_path):
    _pull(tmp_path, "2019-01-10")
    _pull(tmp_path, "2019-01-11")
    pulls, long = load_trends.read_all(tmp_path)
    assert list(long.columns) == load_trends.OBSERVATIONS_COLUMNS
    assert set(long["pull_id"]) == {"2019-01-10", "2019-01-11"}
    assert set(long["entity"]) == {"yape", "plin"}
    assert set(long["series_code"]) == {"GT_YAPE_PLIN"}
    assert len(long) == 2 * 2 * 24                                     # pulls x entities x months
    assert max(long["obs_date"]) == pd.Timestamp("2018-12-01").date()
    assert (long.groupby(["pull_id", "entity", "obs_date"]).size() == 1).all()


def test_loader_dry_run_needs_no_bigquery_credentials(tmp_path):
    _pull(tmp_path)
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS")}
    env["PYTHONPATH"] = "."
    out = subprocess.run(
        [sys.executable, "-m", "src.data_collection.load_trends", "--dry-run", "--dir", str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "--dry-run: no data written" in out.stdout and "48 row(s)" in out.stdout


def test_loader_refuses_a_csv_without_a_sidecar(tmp_path):
    _csv(tmp_path / "2019-01-10_yape_plin.csv", _rows(24))
    with pytest.raises(TrendsFormatError, match="no sidecar"):
        load_trends.read_all(tmp_path)


# --------------------------------------------------------------------------- #
# config registry
# --------------------------------------------------------------------------- #
def test_gt_yape_plin_is_registered_with_kappa_zero_and_log_diff_and_never_fetched_from_bcrp():
    from src.data_collection import config

    assert config.KAPPA_BY_COL["gt_yape_plin"] == 0
    assert config.TRANSFORM_BY_COL["gt_yape_plin"] == "log_diff"
    assert config.COL_NAME_BY_CODE[load_trends.SERIES_CODE] == "gt_yape_plin"
    assert "gt_yape_plin" not in {s["col_name"] for s in config.BCRP_SERIES}
    assert len(config.BCRP_SERIES) == len(config.ALL_SERIES) - 1


# --------------------------------------------------------------------------- #
# T8 -- single-pull rule. dbt is NOT run here (offline suite): these pin the
# artefacts that enforce it, so deleting one fails a test rather than a build
# nobody is watching. The assertions themselves run in `dbt build`.
# --------------------------------------------------------------------------- #
DBT = Path("pagos_dbt")


def test_t8_stg_trends_reads_one_declared_pull_and_the_panel_unions_it():
    import yaml

    sql = (DBT / "models/staging/stg_trends.sql").read_text(encoding="utf-8")
    assert "var('trends_pull_id'" in sql and "where pull_id" in sql
    assert "group by pulls.series_code, pulls.obs_date, pulls.pull_id" in sql
    assert "row_number" not in sql.lower()           # no month-by-month latest-pull de-duplication

    project = yaml.safe_load((DBT / "dbt_project.yml").read_text(encoding="utf-8"))
    declared = project["vars"]["trends_pull_id"]
    assert (trends.RAW_DIR / f"{declared}_yape_plin.csv").exists(), "var points at a pull not in the repo"

    panel = (DBT / "models/marts/monthly_panel.sql").read_text(encoding="utf-8")
    assert "ref('stg_trends')" in panel and "ref('stg_bcrp_observations')" in panel


def test_t8_the_two_singular_tests_exist():
    one = (DBT / "tests/assert_stg_trends_single_pull.sql").read_text(encoding="utf-8")
    assert "count(distinct pull_id) != 1" in one
    two = (DBT / "tests/assert_panel_gt_from_one_pull.sql").read_text(encoding="utf-8")
    assert "ref('monthly_panel')" in two and "ref('stg_trends')" in two


def test_snapshot_version_carries_the_single_trends_pull_and_refuses_two():
    from src.model_training.snapshot import compose_version

    assert compose_version("20261005T000000Z", []) == "20261005T000000Z"
    v = compose_version("20261005T000000Z", ["2026-10-05"])
    assert v == "20261005T000000Z_gt20261005"
    assert v != "20261005T000000Z"        # a 1.6 snapshot of the same BCRP pull is never overwritten
    with pytest.raises(RuntimeError, match="single-pull"):
        compose_version("20261005T000000Z", ["2026-10-05", "2026-10-12"])
