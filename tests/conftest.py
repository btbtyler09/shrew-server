"""Pytest configuration: make `app/` importable as a top-level package
without requiring an editable install."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
