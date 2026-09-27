"""Pytest configuration: make `app/` importable as a top-level package
without requiring an editable install."""

import sys

import pytest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def v0315_env(monkeypatch):
    """The v0.3.x greedy decode ladder, restored the way an operator would
    (README "Decode recipe"). SHREW_RETRY_PP is read at import, so the module
    value is patched alongside the env. Tests of the enforced-retry / re-roll
    rungs use this; the v0.4.0 defaults are pinned in test_decode_knobs."""
    import app.structured_page as sp
    for k, v in (("SHREW_TRY1_TEMP", "0"), ("SHREW_TRY2_TEMP", "0"),
                 ("SHREW_TRY2_ENFORCE", "1"), ("SHREW_ABORT_REROLL", "1"),
                 ("SHREW_RETRY_PP", "0.6")):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sp, "RETRY_PP", 0.6)
