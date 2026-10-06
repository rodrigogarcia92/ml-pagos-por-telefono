"""Freeze the selected model (ens3 / FS3_activity / t3 / w2019) into configs/production/.

    .venv\\Scripts\\python.exe scripts/freeze_production.py
    .venv\\Scripts\\python.exe scripts/freeze_production.py --snapshot data/processed/panel_20261005T000000Z.parquet

Read-only on MLflow and on the snapshot; creates no run. Logic lives in src/forecasting/freeze.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.freeze import main  # noqa: E402

if __name__ == "__main__":
    main()
