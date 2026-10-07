"""Step 6 -- the monthly-refresh workflow: structure, auth, permissions, outcomes. No network."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

PATH = Path(".github/workflows/monthly-refresh.yml")
TEXT = PATH.read_text(encoding="utf-8")
DOC = yaml.safe_load(TEXT)
STEPS = DOC["jobs"]["refresh"]["steps"]


def _on():
    # PyYAML (YAML 1.1) reads the bare key `on` as the boolean True.
    return DOC.get("on", DOC.get(True))


def _uses(prefix: str):
    return [s for s in STEPS if str(s.get("uses", "")).startswith(prefix)]


def test_it_parses_and_runs_on_ubuntu_with_python_311():
    job = DOC["jobs"]["refresh"]
    assert job["runs-on"] == "ubuntu-24.04"
    py, = _uses("actions/setup-python@")
    assert py["with"]["python-version"] == "3.11"
    assert DOC["name"] == "Monthly refresh"


def test_manual_dispatch_and_the_three_monthly_attempts_are_active():
    on = _on()
    assert set(on) == {"workflow_dispatch", "schedule"}
    inp = on["workflow_dispatch"]["inputs"]["skip_etl"]
    assert inp["type"] == "boolean" and inp["default"] is False
    assert on["schedule"] == [{"cron": "50 13 10 * *"}, {"cron": "50 13 20 * *"}, {"cron": "50 13 28 * *"}]


def test_permissions_are_exactly_the_four_requested():
    assert DOC["permissions"] == {"contents": "write", "pull-requests": "write",
                                  "issues": "write", "id-token": "write"}


def test_auth_is_workload_identity_federation_with_no_json_key_anywhere():
    auth, = _uses("google-github-actions/auth@v3")
    assert auth["with"] == {"workload_identity_provider": "${{ secrets.GCP_WIF_PROVIDER }}",
                            "service_account": "${{ secrets.GCP_SERVICE_ACCOUNT }}"}
    assert _uses("google-github-actions/setup-gcloud@")
    assert STEPS.index(auth) < STEPS.index(_uses("google-github-actions/setup-gcloud@")[0])
    low = TEXT.lower()
    assert "credentials_json" not in low and "gcp_sa_key" not in low and "service_account_key" not in low
    assert "private_key" not in low and "keyfile" not in low


def test_the_dbt_profile_is_generated_at_runtime_and_never_committed():
    step = next(s for s in STEPS if s.get("name") == "Write the dbt profile")
    script = step["run"]
    prof = yaml.safe_load(script.split("<<'EOF'\n")[1].split("\nEOF")[0])
    out = prof["pagos_dbt"]["outputs"]["dev"]
    assert prof["pagos_dbt"]["target"] == "dev"
    assert (out["type"], out["method"], out["project"], out["location"], out["dataset"]) == (
        "bigquery", "oauth", "pagos-telefono-26", "US", "staging")
    assert "DBT_PROFILES_DIR=$RUNNER_TEMP/dbt" in script and "DBT_BIN=" in script
    assert not Path("pagos_dbt/profiles.yml").exists()
    assert "profiles.yml" in Path(".gitignore").read_text(encoding="utf-8")
    # the profile name must be the one dbt_project.yml expects
    assert yaml.safe_load(Path("pagos_dbt/dbt_project.yml").read_text(encoding="utf-8"))[
        "profile"] == "pagos_dbt"


def test_two_separate_virtualenvs_with_the_repos_requirement_files():
    runs = "\n".join(s.get("run", "") for s in STEPS)
    assert "python -m venv .venv\n" in runs and "python -m venv .venv-dbt\n" in runs
    assert ".venv/bin/pip install -r requirements-dev.txt" in runs
    assert ".venv-dbt/bin/pip install -r requirements-dbt.txt" in runs
    assert "requirements-dbt.txt" not in runs.split(".venv-dbt/bin/pip")[0].split("python -m venv .venv\n")[1]
    assert "../.venv-dbt/bin/dbt deps" in runs


def test_it_runs_the_orchestrator_with_the_app_environment_and_reacts_to_its_exit_code():
    step = next(s for s in STEPS if s.get("id") == "refresh")
    assert ".venv/bin/python scripts/monthly_refresh.py" in step["run"]
    assert step["env"] == {"SKIP_ETL": "${{ inputs.skip_etl }}"}
    assert "--from-stage snapshot" in step["run"]          # skip_etl still needs a snapshot in CI
    assert "rc=$rc" in step["run"] and "set +e" in step["run"]


def test_a_new_month_opens_a_titled_pull_request_with_the_right_files():
    pr_step, = _uses("peter-evans/create-pull-request@v8")
    assert pr_step["if"] == "steps.summary.outputs.outcome == 'forecast'"
    w = pr_step["with"]
    assert w["title"] == "forecast: ${{ steps.summary.outputs.target_month }}"
    assert "data/raw/bcrp/*.json" in w["add-paths"] and "forecasts/**" in w["add-paths"]
    assert w["body-path"].endswith("pr_body.md")
    body = next(s for s in STEPS if s.get("name") == "Compose the pull request body")["run"]
    for needle in ("point_forecast", "band_80", "band_90", "monitor.json", "Drift monitor"):
        assert needle in body


def test_no_new_month_opens_nothing():
    for s in STEPS:
        if "pull-request" in str(s.get("uses", "")) or "gh issue create" in s.get("run", ""):
            assert "if" in s                                   # every opener is conditional
    pr_step, = _uses("peter-evans/create-pull-request@v8")
    assert "outcome == 'forecast'" in pr_step["if"] and "no_new_month" not in pr_step["if"]


def test_drift_alert_opens_or_comments_on_an_issue_labelled_drift_alert():
    step = next(s for s in STEPS if s.get("name", "").startswith("Open (or comment on) the drift-alert"))
    assert step["if"] == "steps.refresh.outputs.rc == '2'"
    assert "gh issue list --label drift-alert --state open" in step["run"]
    assert "gh issue comment" in step["run"] and "gh issue create --label drift-alert" in step["run"]


def test_a_failed_stage_opens_a_refresh_failed_issue_then_fails_the_job():
    names = [s.get("name") for s in STEPS]
    i = names.index("Open a refresh-failed issue")
    j = names.index("Fail the job if a stage failed")
    assert i < j == len(STEPS) - 1
    opener, fail = STEPS[i], STEPS[j]
    assert "steps.refresh.outputs.failed == 'true'" in opener["if"]
    assert opener["if"] == fail["if"]
    assert "gh issue create --label refresh-failed" in opener["run"]
    assert ".error.stage" in opener["run"] and ".error.stderr_tail" in opener["run"]
    assert fail["run"].rstrip().endswith("exit 1")
    refresh = next(s for s in STEPS if s.get("id") == "refresh")
    assert '"$rc" -ne 0' in refresh["run"] and '"$rc" -ne 2' in refresh["run"]


def test_labels_are_created_before_they_are_used():
    names = [s.get("name") for s in STEPS]
    lab = next(s for s in STEPS if s.get("name") == "Make sure the labels exist")
    for label in ("forecast", "drift-alert", "refresh-failed"):
        assert f"gh label create {label}" in lab["run"]
    assert names.index("Make sure the labels exist") < names.index("Open the forecast pull request")


def test_gh_steps_use_the_job_token_and_nothing_else_is_a_secret():
    for s in STEPS:
        if "gh " in s.get("run", ""):
            assert s["env"]["GH_TOKEN"] == "${{ github.token }}"
    import re
    assert set(re.findall(r"secrets\.([A-Z_]+)", TEXT)) == {"GCP_WIF_PROVIDER", "GCP_SERVICE_ACCOUNT"}


def test_the_existing_ci_workflow_is_untouched_and_still_secret_free():
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "secrets." not in ci and "id-token" not in ci


@pytest.mark.skipif(shutil.which("actionlint") is None, reason="actionlint not installed")
def test_actionlint_is_clean():
    r = subprocess.run(["actionlint", str(PATH)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
