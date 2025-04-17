import pytest

from settings import Config
from utility_route_planner.util.write import reset_geopackage


@pytest.fixture
def setup_mcda_lcpa_testing(monkeypatch):
    reset_geopackage(Config.PATH_GEOPACKAGE_LCPA_OUTPUT)
    reset_geopackage(Config.PATH_GEOPACKAGE_MCDA_OUTPUT, truncate=False)
    monkeypatch.setattr(Config, "DEBUG", True)
