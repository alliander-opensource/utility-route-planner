# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import math
import geopandas as gpd
import geopandas.testing
import numpy as np
import pytest
import shapely

from settings import Config
from utility_route_planner.models.mcda.load_mcda_preset import load_preset
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_grid_builder import (
    HexagonGridBuilder,
)


@pytest.fixture()
def preprocessed_vectors() -> dict[str, gpd.GeoDataFrame]:
    return {"test": gpd.GeoDataFrame()}


@pytest.fixture()
def raster_groups() -> dict[str, str]:
    preset = load_preset(
        Config.RASTER_PRESET_NAME_BENCHMARK,
        Config.PYTEST_PATH_GEOPACKAGE_MCDA,
        gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA).iloc[0].geometry,
    )
    return {criteria_key: criteria.group for criteria_key, criteria in preset.criteria.items()}


@pytest.fixture()
def grid_constructor(
    raster_groups: dict[str, str], preprocessed_vectors: dict[str, gpd.GeoDataFrame]
) -> HexagonGridBuilder:
    hexagon_size = 0.5
    block_size = 512
    return HexagonGridBuilder(raster_groups, preprocessed_vectors, hexagon_size, block_size)


class TestConstructHexagonalGridForBoundingBox:
    @pytest.fixture()
    def square_project_area(self) -> shapely.Polygon:
        return shapely.Polygon(
            [
                shapely.Point(0, 0),
                shapely.Point(5, 0),
                shapely.Point(5, 5),
                shapely.Point(0, 5),
                shapely.Point(0, 0),
            ]
        )

    @pytest.fixture()
    def triangular_project_area(self) -> shapely.Polygon:
        return shapely.Polygon(
            [
                shapely.Point(0, 0),
                shapely.Point(4, 0),
                shapely.Point(2, 4),
                shapely.Point(0, 0),
            ]
        )

    def test_square_project_area(self, grid_constructor: HexagonGridBuilder, square_project_area: shapely.Polygon):
        result = grid_constructor.construct_hexagonal_grid_for_bounding_box(square_project_area)

        coordinates = result.geometry.get_coordinates()
        x_coordinates = coordinates["x"].values.reshape(6, 7)
        y_coordinates = coordinates["y"].values.reshape(6, 7)

        hexagon_size = grid_constructor.hexagon_size
        self.check_grid_bounding_box(hexagon_size, square_project_area, x_coordinates, y_coordinates)
        self.check_grid_spacing(hexagon_size, x_coordinates, y_coordinates)

    def test_triangular_project_area(
        self, grid_constructor: HexagonGridBuilder, triangular_project_area: shapely.Polygon
    ):
        result = grid_constructor.construct_hexagonal_grid_for_bounding_box(triangular_project_area)
        coordinates = result.geometry.get_coordinates()
        x_coordinates = coordinates["x"].values.reshape(5, 6)
        y_coordinates = coordinates["y"].values.reshape(5, 6)

        self.check_grid_spacing(grid_constructor.hexagon_size, x_coordinates, y_coordinates)

        hexagon_size = grid_constructor.hexagon_size
        self.check_grid_bounding_box(hexagon_size, triangular_project_area, x_coordinates, y_coordinates)
        self.check_grid_spacing(hexagon_size, x_coordinates, y_coordinates)

    @staticmethod
    def check_grid_bounding_box(
        hexagon_size: float, project_area: shapely.Polygon, x_coordinates: np.ndarray, y_coordinates: np.ndarray
    ):
        """
        Verifies that all coordinates are within or on the border of the project area polygon bounding box. At this stage,
        the grid should be equal to the square equal to the bounding box, independent of the shape of the project area polygon.
        """
        exptected_x_min, exptected_y_min, exptected_x_max, exptected_y_max = project_area.bounds

        assert pytest.approx(exptected_x_min) == x_coordinates.min()
        assert exptected_x_max >= x_coordinates.max() >= (exptected_x_max - hexagon_size)
        assert pytest.approx(exptected_y_min) == y_coordinates.min()
        assert exptected_y_max >= y_coordinates.max() >= (exptected_y_max - hexagon_size)

    @staticmethod
    def check_grid_spacing(hexagon_size: float, x_coordinates: np.ndarray, y_coordinates: np.ndarray):
        # Set expected spacing between hexagon points based on known equations.
        expected_horizontal_spacing = -3 / 2 * hexagon_size
        expected_vertical_spacing = -math.sqrt(3) * hexagon_size

        # Verify that spacing between x-coordinates of hexagon center points equals the expected horizontal spacing
        x_spacing = np.diff(x_coordinates, axis=1)
        assert all(x_space == pytest.approx(expected_horizontal_spacing, abs=1e-4) for x_space in x_spacing)

        # Verify that spacing between y-coordinates of hexagon center points equals the expected vertical spacing
        y_vertical_spacing = np.diff(y_coordinates, axis=0)
        assert all(y_space == pytest.approx(expected_vertical_spacing, abs=1e-4) for y_space in y_vertical_spacing)

        # Verify that y-coordinates of horizontally neighbouring hexagon center points are 1/2 of the expected vertical spacing
        # separated from each other. This is due to alternating offset of neighbouring hexagon points.
        y_horizontal_spacing = np.diff(y_coordinates, axis=1)
        expected_y_spacing = abs(expected_vertical_spacing) / 2
        assert all(y_space == pytest.approx(expected_y_spacing, abs=1e-4) for y_space in np.abs(y_horizontal_spacing))

        # As every even column is offset by 1/2 the hexagon height, the horizontal spacing should always be negative. For
        # all odd columns, the sign should be positive
        signs = np.sign(y_horizontal_spacing)
        assert all(sign == -1 for sign in signs[:, ::2].flatten())
        assert all(sign == 1 for sign in signs[:, 1::2].flatten())


class TestAssignSuitabilityValuesToGrid:
    def test_no_overlapping_points(self, grid_constructor: HexagonGridBuilder):
        """
        Verify that all suitability values remain intact in case no points are overlapping. Only points in group c
        should have a suitability value which equals the max node suitability value.
        """
        points_on_grid = gpd.GeoDataFrame(
            data=[[1, "a", 20.0], [2, "b", 30.0], [3, "c", 40.0], [4, "a", 50.0], [5, "b", 60.0], [6, "c", 70.0]],
            columns=["node_id", "group", "suitability_value"],
            geometry=[
                shapely.Point(0, 0),
                shapely.Point(1, 1),
                shapely.Point(2, 2),
                shapely.Point(3, 3),
                shapely.Point(4, 4),
                shapely.Point(5, 5),
            ],
            crs=Config.CRS,
        ).set_index("node_id")

        result = grid_constructor.assign_suitability_values_to_block(points_on_grid)
        expected_suitability_values = (
            gpd.GeoDataFrame(
                data=[
                    [1, 20],
                    [2, 30],
                    [3, Config.MAX_NODE_SUITABILITY_VALUE],
                    [4, 50],
                    [5, 60],
                    [6, Config.MAX_NODE_SUITABILITY_VALUE],
                ],
                columns=["node_id", "suitability_value"],
                geometry=[
                    shapely.Point(0, 0),
                    shapely.Point(1, 1),
                    shapely.Point(2, 2),
                    shapely.Point(3, 3),
                    shapely.Point(4, 4),
                    shapely.Point(5, 5),
                ],
                crs=Config.CRS,
            )
            .set_index("node_id")
            .astype({"suitability_value": np.int16})
        )

        gpd.testing.assert_geodataframe_equal(expected_suitability_values, result.sort_index())

    def test_overlapping_points_group_a(self, grid_constructor: HexagonGridBuilder):
        """
        Verify that for overlapping points in group a, the max value is used as suitability value
        that point.
        """
        points_on_grid = gpd.GeoDataFrame(
            data=[[1, "a", 20.0], [1, "a", 30.0]],
            columns=["node_id", "group", "suitability_value"],
            geometry=[shapely.Point(0, 0), shapely.Point(0, 0)],
            crs=Config.CRS,
        ).set_index("node_id")

        result = grid_constructor.assign_suitability_values_to_block(points_on_grid)

        expected_suitability_values = (
            gpd.GeoDataFrame(
                data=[[1, 30.0]],
                columns=["node_id", "suitability_value"],
                geometry=[shapely.Point(0, 0)],
                crs=Config.CRS,
            )
            .set_index("node_id")
            .astype({"suitability_value": np.int16})
        )

        gpd.testing.assert_geodataframe_equal(expected_suitability_values, result)

    def test_overlapping_points_group_b(self, grid_constructor: HexagonGridBuilder):
        """
        Verify that for overlapping points in group b, values for all points are summed to
        compute the suitability value
        """
        points_on_grid = gpd.GeoDataFrame(
            data=[[1, "b", 50.0], [1, "b", 40.0]],
            columns=["node_id", "group", "suitability_value"],
            geometry=[shapely.Point(0, 0), shapely.Point(0, 0)],
            crs=Config.CRS,
        ).set_index("node_id")

        result = grid_constructor.assign_suitability_values_to_block(points_on_grid)

        expected_suitability_values = (
            gpd.GeoDataFrame(
                data=[[1, 90]], columns=["node_id", "suitability_value"], geometry=[shapely.Point(0, 0)], crs=Config.CRS
            )
            .set_index("node_id")
            .astype({"suitability_value": np.int16})
        )

        gpd.testing.assert_geodataframe_equal(expected_suitability_values, result)

    def test_overlapping_points_group_c(self, grid_constructor: HexagonGridBuilder):
        """
        Verify that for overlapping points in group c, the suitability value for the respective nodes is always
        set to the max node suitability value.
        """
        points_on_grid = gpd.GeoDataFrame(
            data=[[1, "c", 100.0], [1, "b", 100.0]],
            columns=["node_id", "group", "suitability_value"],
            geometry=[shapely.Point(0, 0), shapely.Point(0, 0)],
            crs=Config.CRS,
        ).set_index("node_id")

        result = grid_constructor.assign_suitability_values_to_block(points_on_grid)

        expected_suitability_values = (
            gpd.GeoDataFrame(
                data=[[1, Config.MAX_NODE_SUITABILITY_VALUE]],
                columns=["node_id", "suitability_value"],
                geometry=[shapely.Point(0, 0)],
                crs=Config.CRS,
            )
            .set_index("node_id")
            .astype({"suitability_value": np.int16})
        )

        gpd.testing.assert_geodataframe_equal(expected_suitability_values, result)

    def test_sum_overlapping_group_a_and_b(self, grid_constructor: HexagonGridBuilder):
        """
        Verify that when nodes from group a and b intersect, the values are summed. Suitability values of
        all nodes in these groups that do not intersect should remain intact.
        """
        points_on_grid = gpd.GeoDataFrame(
            data=[
                # Node 1 intersects with two groups, these values must be summed
                [1, "a", 20.0],
                [1, "b", 30.0],
                # Node 2 & 3 do not intersect. The suitability values must remain intact
                [2, "a", 40.0],
                [3, "b", 50.0],
            ],
            columns=["node_id", "group", "suitability_value"],
            geometry=[shapely.Point(0, 0), shapely.Point(0, 0), shapely.Point(1, 1), shapely.Point(2, 2)],
            crs=Config.CRS,
        ).set_index("node_id")

        result = grid_constructor.assign_suitability_values_to_block(points_on_grid)

        expected_suitability_values = (
            gpd.GeoDataFrame(
                data=[[1, 50], [2, 40], [3, 50]],
                columns=["node_id", "suitability_value"],
                geometry=[shapely.Point(0, 0), shapely.Point(1, 1), shapely.Point(2, 2)],
                crs=Config.CRS,
            )
            .set_index("node_id")
            .astype({"suitability_value": np.int16})
        )

        gpd.testing.assert_geodataframe_equal(expected_suitability_values, result.sort_index())

    @pytest.mark.parametrize("group", ["a", "b"])
    def test_group_a_or_b_filled_while_other_empty(self, group: str, grid_constructor: HexagonGridBuilder):
        """
        In case either group a or b is filled while the other group is empty, the assigned suitability
        values for each point should be equal to that of the filled group.
        """
        points_on_grid = gpd.GeoDataFrame(
            data=[[1, group, 20.0], [2, group, 40.0]],
            columns=["node_id", "group", "suitability_value"],
            geometry=[shapely.Point(0, 0), shapely.Point(1, 1)],
            crs=Config.CRS,
        ).set_index("node_id")

        result = grid_constructor.assign_suitability_values_to_block(points_on_grid)

        expected_suitability_values = (
            gpd.GeoDataFrame(
                data=[[1, 20], [2, 40]],
                columns=["node_id", "suitability_value"],
                geometry=[shapely.Point(0, 0), shapely.Point(1, 1)],
                crs=Config.CRS,
            )
            .set_index("node_id")
            .astype({"suitability_value": np.int16})
        )

        gpd.testing.assert_geodataframe_equal(expected_suitability_values, result.sort_index())


@pytest.mark.skip(reason="TODO: Add new coordinates after bug fix")
class TestCartesianToAxialConversion:
    def test_conversion(self, grid_constructor: HexagonGridBuilder):
        center_points = gpd.GeoDataFrame(
            geometry=[
                shapely.Point(174966.804, 451064.681),
                shapely.Point(174967.554, 451065.114),
                shapely.Point(174968.304, 451064.681),
                shapely.Point(174967.554, 451064.248),
                shapely.Point(174966.804, 451063.815),
                shapely.Point(174967.554, 451063.382),
                shapely.Point(174968.304, 451063.815),
            ]
        )
        xgrid_result, ygrid_result = grid_constructor.convert_cartesian_coordinates_to_axial(center_points)

        #                           -1         0         +1          center(0)  -1         0         +1
        expected_xgrid = np.array([[-233289], [-233290], [-233291], [-233290], [-233289], [-233290], [-233291]])

        #                           0        +1        +1        center(0) -1        -1        0
        expected_ygrid = np.array([[637489], [637490], [637490], [637489], [637488], [637488], [637489]])

        assert all(expected_xgrid == xgrid_result)
        assert all(expected_ygrid == ygrid_result)
