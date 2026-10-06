"""Monthly refresh: fetch -> load -> dbt -> snapshot -> predict -> monitor.

    .venv\\Scripts\\python.exe scripts/monthly_refresh.py --dry-run --skip-etl
    .venv\\Scripts\\python.exe scripts/monthly_refresh.py --skip-etl
    .venv\\Scripts\\python.exe scripts/monthly_refresh.py

Exit codes: 0 done (or no new month), 2 done but drift ALERT, 1 a stage failed.
Logic and the full description: src/forecasting/refresh.py. Run from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.refresh import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
