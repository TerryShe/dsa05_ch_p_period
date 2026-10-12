from dataclasses import replace

import pytest

from server.catalog import Catalog
from server.registry import EXP011
from server.settings import load_settings


@pytest.fixture(scope="session")
def settings():
    s = load_settings()
    if not (s.outputs / EXP011.outputs / EXP011.analysis).exists():
        pytest.skip("formal EXP-011 outputs are not available")
    return s


@pytest.fixture(scope="session")
def results(settings):
    return Catalog(settings).get("exp011")


@pytest.fixture
def before_report(settings):
    """Settings whose gate ref predates the EXP-011 report (4f979d9, EXP-011b approval)."""
    return replace(settings, gate_ref="4f979d9")
