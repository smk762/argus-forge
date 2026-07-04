from __future__ import annotations

import argus_forge


def test_version_is_exposed() -> None:
    assert isinstance(argus_forge.__version__, str)
    assert argus_forge.__version__
