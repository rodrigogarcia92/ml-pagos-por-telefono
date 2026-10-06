"""The one-time holdout, behind guards (marker in plan section 11, --confirm, not yet run).

    .venv\\Scripts\\python.exe scripts/run_holdout.py              # prints the plan, exits 0
    .venv\\Scripts\\python.exe scripts/run_holdout.py --confirm    # the real run, once, ever

Needs the MLflow server (scripts/start_mlflow.ps1) for the real run. Logic: src/forecasting/holdout.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.holdout import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
