"""Shim. The implementation lives in src/model_training/report.py.

    python -m src.model_training.report --target t2 --horizon 1     <- preferred
    python scripts/compare_runs.py --target t2 --horizon 1          <- also works

Running a script by path puts its OWN directory on sys.path, not the project
root, so `import src.model_training` fails -- the same trap that broke bare
`pytest` before pyproject.toml fixed that case. Three lines here make the
by-path form work too, rather than leaving a command that looks reasonable and
is not.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_training.report import main  # noqa: E402

if __name__ == "__main__":
    main()
