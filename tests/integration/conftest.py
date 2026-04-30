# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import pytest
import geopandas as gpd
import shapely

from settings import Config
from utility_route_planner.util.write import reset_geopackage


@pytest.fixture
def setup_mcda_lcpa_testing(monkeypatch):
    reset_geopackage(Config.PATH_GEOPACKAGE_LCPA_OUTPUT)
    reset_geopackage(Config.PATH_GEOPACKAGE_MCDA_OUTPUT, truncate=False)
    monkeypatch.setattr(Config, "DEBUG", True)


@pytest.fixture
def single_criterion_vectors() -> Callable:
    def __vectors(max_value: int, min_value: int, no_data: int) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            data=[
                # These layers all overlap each other.
                [1, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
                [10, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
                [10, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
                [1, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
                # One larger partly overlapping polygon with a unique value.
                [
                    5,
                    shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]]).buffer(
                        50
                    ),
                ],
                # These values should be reset to the min/max of the intermediate raster values
                [
                    min_value - 1000,
                    shapely.Polygon([[175091, 450919], [175091, 450911], [175105, 450911], [175091, 450919]]),
                ],
                [
                    max_value + 1000,
                    shapely.Polygon([[175012, 450920], [175011, 450907], [175019, 450906], [175012, 450920]]),
                ],
                [no_data, shapely.Polygon([[174917, 450965], [174937, 450962], [174916, 450952], [174917, 450965]])],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "geometry"],
        )

    return __vectors


@pytest.fixture
def multi_criteria_vectors() -> Callable:
    def __vectors(max_value: int, min_value: int) -> list[tuple[str, gpd.GeoDataFrame, str]]:
        # 4 rasters:
        # 1. group a - partial overlap
        criterion_a_1 = gpd.GeoDataFrame(
            data=[
                # These layers all overlap each other.
                [1, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
                [10, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
                # One larger partly overlapping polygon with a unique value.
                [
                    5,
                    shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]]).buffer(
                        50
                    ),
                ],
                # These values should be reset to the min/max of the intermediate raster values
                [
                    min_value - 1000,
                    shapely.Polygon([[175091, 450919], [175091, 450911], [175105, 450911], [175091, 450919]]),
                ],
                [
                    max_value + 1000,
                    shapely.Polygon([[175012, 450920], [175011, 450907], [175019, 450906], [175012, 450920]]),
                ],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "geometry"],
        )
        # 2. group a - partial overlap
        criterion_a_2 = gpd.GeoDataFrame(
            data=[
                # Overlaps criterion a 1 with a higher value
                [50, shapely.Polygon([[174797, 451107], [174944, 451090], [174807, 451129], [174797, 451107]])],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "geometry"],
        )
        # 3. group b - partial overlap
        criterion_b_1 = gpd.GeoDataFrame(
            data=[
                # Overlaps criterion a 1 with a higher value
                [20, shapely.Point([174813.3770, 451113.8469]).buffer(2)],
                [1000, shapely.Point([174870.46, 451051.07])],
                [-20, shapely.Point([175013.310, 450910.294])],
                [-1, shapely.Polygon([[175087, 450911], [175107, 450912], [175087, 450915], [175087, 450911]])],
                # Overlaps criterion a 1 with the same value but signed.
                [-5, shapely.Polygon([[174830, 451074], [174842, 451069], [174831, 451061], [174830, 451074]])],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "geometry"],
        )
        # 4. group b - overlaps criterion b1 and a1
        criterion_b_2 = gpd.GeoDataFrame(
            data=[
                [15, shapely.Polygon([[175096, 450908], [175089, 450908], [175091, 450921], [175096, 450908]])],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "geometry"],
        )
        # 5. group b - tree on the edge of the project area. The buffer exceeds the project area boundary.
        criterion_b_3 = gpd.GeoDataFrame(
            data=[[10, shapely.Point(174741.950, 451113.084).buffer(10)]],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "geometry"],
        )
        # 6. group c - overlapping a1
        criterion_c_1 = gpd.GeoDataFrame(
            data=[
                [1, shapely.Polygon([[174729, 451158], [174940, 451115], [174841, 451195], [174729, 451158]])],
                [10, shapely.Polygon([[174915, 451128], [174924, 451135], [174926, 451109], [174915, 451128]])],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "geometry"],
        )
        # 7. group c - overlapping b1 and c1
        criterion_c_2 = gpd.GeoDataFrame(
            data=[
                [1, shapely.Polygon([[175090, 450906], [175103, 450905], [175096, 450918], [175090, 450906]])],
                [39, shapely.Polygon([[174811, 451226], [174834, 451155], [174909, 451174], [174811, 451226]])],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "geometry"],
        )

        return [
            ("criterion_a1", "a", criterion_a_1),
            ("criterion_a2", "a", criterion_a_2),
            ("criterion_b1", "b", criterion_b_1),
            ("criterion_b2", "b", criterion_b_2),
            ("criterion_b3", "b", criterion_b_3),
            ("criterion_c1", "c", criterion_c_1),
            ("criterion_c2", "c", criterion_c_2),
        ]

    return __vectors
