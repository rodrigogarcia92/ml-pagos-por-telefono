"""One-off smoke test for the MLflow stack.

Proves server + SQLite backend + GCS artifact root + Model Registry + aliases all
work BEFORE any real modelling code exists, so a failure here is a setup problem
and not a modelling problem.

    python scripts/smoke_mlflow.py

Then check the UI and the GCS bucket, and delete the experiment and
the registered model. Nothing here touches the real experiments.

MLflow 3.x: models are logged with `name=`, and promotion uses registry ALIASES
(`@production`) — stages were removed in 3.0.
"""

import tempfile
from pathlib import Path

import mlflow
import numpy as np
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.linear_model import Ridge

load_dotenv()

EXPERIMENT = "26.1__smoke"
MODEL_NAME = "smoke_ridge"

print(f"tracking uri : {mlflow.get_tracking_uri()}")
print(f"mlflow       : {mlflow.__version__}")

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 3))
y = X @ np.array([1.0, -0.5, 0.2]) + rng.normal(scale=0.1, size=40)

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="smoke__parent") as parent:
    # A representative subset of the ten mandatory tags (plan §8.2)
    mlflow.set_tags({
        "protocol_version": "1.2",
        "stage": "cv",
        "model_family": "ridge",
        "window": "w2019",
        "data_version": "smoke",
    })
    mlflow.log_params({"alpha": 1.0, "min_train": 36, "purge": 0, "n_features": 3})

    model = Ridge(alpha=1.0).fit(X, y)

    # Parent-only artifact (§8.5) — stands in for tuning_results.csv
    with tempfile.TemporaryDirectory() as d:
        p = Path(d, "tuning_results.csv")
        p.write_text("alpha,mase_inner\n0.1,0.94\n1.0,0.87\n10.0,0.91\n")
        mlflow.log_artifact(str(p))

    # Children: metrics and params ONLY, no artifacts (§8.5)
    for fold in range(3):
        with mlflow.start_run(run_name=f"fold_{fold:02d}", nested=True):
            mlflow.log_metrics({"mase": 0.90 - 0.01 * fold, "skill_h": 0.10 + 0.01 * fold})

    mlflow.log_metrics({"mase_mean": 0.89, "mase_std": 0.01})

    info = mlflow.sklearn.log_model(
        model, name="model", registered_model_name=MODEL_NAME
    )

print(f"\nparent run_id: {parent.info.run_id}")

# --- Registry: this is the check that proves the SQLite backend is doing its job.
# A file-based store cannot do any of the next four lines.
client = MlflowClient()
version = client.get_latest_versions(MODEL_NAME)[0].version
client.set_registered_model_alias(MODEL_NAME, "production", version)
resolved = client.get_model_version_by_alias(MODEL_NAME, "production")
print(f"registered   : {MODEL_NAME} v{version}, alias @production -> v{resolved.version}")

# --- Round-trip: load back through the alias, exactly as FastAPI will at step 11.
loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
print(f"round-trip   : {loaded.predict(X[:2])}")

print("\nNow verify in the UI and in GCS.")
print("  gcloud storage ls -r gs://pagos-telefono-26-mlflow/")
