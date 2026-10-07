"""Google Trends is opt-in in dbt: the production build must not need raw.trends_observations.

Static checks on the dbt project (dbt itself lives in its own environment, so CI's application
tests cannot parse it). The real proof is `dbt ls --select stg_trends+` selecting nothing.
"""
import re
from pathlib import Path

import yaml

DBT = Path(__file__).resolve().parents[1] / "pagos_dbt"


def _text(rel: str) -> str:
    return (DBT / rel).read_text(encoding="utf-8")


def test_enable_trends_defaults_to_off():
    project = yaml.safe_load(_text("dbt_project.yml"))
    assert project["vars"]["enable_trends"] is False


def test_trends_only_nodes_are_gated_by_the_var():
    for rel in ("models/staging/stg_trends.sql",
                "tests/assert_stg_trends_single_pull.sql",
                "tests/assert_panel_gt_from_one_pull.sql"):
        assert "enabled=var('enable_trends', false)" in _text(rel), rel


def test_every_other_use_of_stg_trends_is_inside_the_gate():
    for rel in ("models/marts/monthly_panel.sql", "models/marts/series_coverage.sql"):
        sql = _text(rel)
        for m in re.finditer(r"ref\('stg_trends'\)", sql):
            before = sql[:m.start()]
            assert before.rfind("{%- if") > before.rfind("{%- endif"), f"{rel}: ungated stg_trends"


def test_trends_yaml_pieces_are_gated():
    gate = "{{ var('enable_trends', false) }}"
    assert gate in _text("models/staging/_sources.yml")
    assert _text("models/staging/_staging__models.yml").count(gate) == 2
    assert gate in _text("models/marts/_marts__models.yml")


def test_monthly_refresh_script_does_not_mention_trends():
    refresh = Path(__file__).resolve().parents[1] / "scripts" / "monthly_refresh.py"
    assert "trends" not in refresh.read_text(encoding="utf-8").lower()
