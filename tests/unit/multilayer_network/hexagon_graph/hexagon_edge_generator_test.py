#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
import math

import geopandas as gpd
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest
import shapely

from settings import Config
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_edge_generator import HexagonEdgeGenerator
from utility_route_planner.util.write import write_results_to_geopackage


class TestHexagonEdgeGenerator:
    @pytest.fixture()
    def hexagon_size(self) -> int:
        return 4

    @pytest.fixture()
    def hexagonal_grid(self) -> pl.DataFrame:
        """
        Hexagonal grid generated with hexagonsize=4
        """
        return pl.DataFrame(
            data=[
                [0, 19, -29162, 79691],
                [1, 9, -29160, 79690],
                [2, 76, -29160, 79692],
                [3, 76, -29162, 79692],
                [4, 126, -29161, 79693],
                [5, 126, -29161, 79692],
                [6, 126, -29162, 79693],
                [7, 76, -29160, 79691],
                [8, 86, -29161, 79691],
                [9, 126, -29163, 79694],
                [10, 126, -29163, 79693],
                [11, 76, -29165, 79693],
                [12, 19, -29164, 79692],
                [13, 76, -29163, 79692],
                [14, 76, -29163, 79696],
                [15, 76, -29163, 79697],
                [16, 76, -29163, 79695],
                [17, 19, 76, -29160, 79694],
                [18, 76, 76, -29160, 79695],
                [19, 126, -29162, 79694],
                [20, 126, -29160, 79693],
                [21, 76, -29161, 79694],
                [22, 76, -29159, 79693],
            ],
            schema={
                "node_id": pl.Int32,
                "suitability_value": pl.Int16,
                "q": pl.Int32,
                "r": pl.Int32,
            },
        )

    @pytest.fixture()
    def block(self):
        return pl.DataFrame(
            data=[
                [0, 19, -29162, 79691],
                [1, 9, -29160, 79690],
                [2, 76, -29160, 79692],
                [3, 76, -29162, 79692],
                [4, 126, -29161, 79693],
                [5, 126, -29161, 79692],
                [6, 126, -29162, 79693],
                [7, 76, -29160, 79691],
                [8, 86, -29161, 79691],
            ],
            schema={
                "node_id": pl.Int32,
                "suitability_value": pl.Int16,
                "q": pl.Int32,
                "r": pl.Int32,
            },
        )

    @pytest.fixture()
    def previous_edge_nodes_current_row(self) -> pl.DataFrame:
        return pl.DataFrame(
            data=[
                [9, 126, -29163, 79694],
                [10, 126, -29163, 79693],
                [11, 76, -29165, 79693],
                [12, 19, -29164, 79692],
                [13, 76, -29163, 79692],
            ],
            schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32},
        )

    @pytest.fixture()
    def relevant_edge_nodes_previous_row(self) -> pl.DataFrame:
        return pl.DataFrame(
            data=[
                [14, 76, -29163, 79696],
                [15, 76, -29163, 79697],
                [16, 76, -29163, 79695],
                [17, 19, 76, -29160, 79694],
                [18, 76, 76, -29160, 79695],
                [19, 126, -29162, 79694],
                [20, 126, -29160, 79693],
                [21, 76, -29161, 79694],
                [22, 76, -29159, 79693],
            ],
            schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32},
        )

    def test_generate_edges_without_previous_blocks(
        self, hexagonal_grid: pl.DataFrame, block: pl.DataFrame, hexagon_size: int, debug: bool = False
    ):
        """
        Test that verifies that all edges are present, have correct edge weights and length when generating edges
        for a block without having previous edges. This means only edges within the block itself are expected.

        This test can be understood most easily by setting debug=True and inspecting the results in QGis.
        """

        empty_previous_edges = pl.DataFrame(
            schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32}
        )
        nodes_to_check = pl.concat([block, empty_previous_edges])
        generator = HexagonEdgeGenerator()
        result_edges = generator.generate(block, nodes_to_check)

        expected_edges = pl.DataFrame(
            data=[
                [5, 6, 252],
                [0, 8, 105],
                [0, 3, 95],
                [4, 6, 252],
                [1, 7, 85],
                [8, 3, 162],
                [3, 5, 202],
                [7, 8, 162],
                [3, 6, 202],
                [1, 8, 95],
                [5, 4, 252],
                [2, 5, 202],
                [2, 4, 202],
                [7, 5, 202],
                [7, 2, 152],
                [8, 5, 212],
            ],
            schema={"source_node": pl.Int32, "target_node": pl.Int32, "weight": pl.Int16},
        )
        assert_frame_equal(expected_edges, result_edges, check_row_order=False)

        hexagon_points, edges_linestrings = self.convert_nodes_and_edges_to_gdfs(
            hexagonal_grid, result_edges, hexagon_size=hexagon_size
        )
        assert all([length == pytest.approx(6.93, abs=0.01) for length in edges_linestrings.length])

        if debug:
            self.write_debug(hexagon_points, edges_linestrings, suffix="no_previous_nodes")

    def test_generate_edges_with_single_previous_block(
        self,
        hexagonal_grid: pl.DataFrame,
        block: pl.DataFrame,
        previous_edge_nodes_current_row: pl.DataFrame,
        hexagon_size: int,
        debug: bool = False,
    ):
        """
        Test that verifies that all edges are present, have correct edge weights and length when generating edges
        for a block when having only the relevant nodes in the block to check for cross-block edges. This mimics the
        situation when processing the next block on the first row of the grid.

        This test can be understood most easily by setting debug=True and inspecting the results in QGis.
        """
        nodes_to_check = pl.concat([block, previous_edge_nodes_current_row])

        generator = HexagonEdgeGenerator()
        result_edges = generator.generate(block, nodes_to_check)

        expected_edges = pl.DataFrame(
            data=[
                [3, 10, 202],
                [7, 5, 202],
                [2, 4, 202],
                [3, 6, 202],
                [3, 13, 152],
                [2, 5, 202],
                [3, 5, 202],
                [5, 4, 252],
                [4, 6, 252],
                [7, 2, 152],
                [0, 8, 105],
                [1, 8, 95],
                [8, 5, 212],
                [6, 10, 252],
                [6, 9, 252],
                [0, 13, 95],
                [7, 8, 162],
                [5, 6, 252],
                [0, 3, 95],
                [8, 3, 162],
                [1, 7, 85],
            ],
            schema={"source_node": pl.Int32, "target_node": pl.Int32, "weight": pl.Int16},
        )
        assert_frame_equal(expected_edges, result_edges, check_row_order=False)

        hexagon_points, edges_linestrings = self.convert_nodes_and_edges_to_gdfs(
            hexagonal_grid, result_edges, hexagon_size=hexagon_size
        )
        assert all([length == pytest.approx(6.93, abs=0.01) for length in edges_linestrings.length])

        if debug:
            self.write_debug(hexagon_points, edges_linestrings, suffix="previous_block_only")

    def tests_generate_edges_with_only_previous_row(
        self,
        hexagonal_grid: pl.DataFrame,
        block: pl.DataFrame,
        relevant_edge_nodes_previous_row: pl.DataFrame,
        hexagon_size: int,
        debug: bool = False,
    ):
        """
        Test that verifies that all edges are present, have correct edge weights and length when generating edges
        for a block when having only the relevant nodes in the previous row to check for cross-block edges. This mimics
        the situation when entering a new row when constructing the grid.

        This test can be understood most easily by setting debug=True and inspecting the results in QGis.
        """
        nodes_to_check = pl.concat([block, relevant_edge_nodes_previous_row])

        generator = HexagonEdgeGenerator()
        result_edges = generator.generate(block, nodes_to_check)

        expected_edges = pl.DataFrame(
            data=[
                [2, 20, 202],
                [7, 5, 202],
                [2, 4, 202],
                [3, 6, 202],
                [4, 19, 252],
                [2, 5, 202],
                [3, 5, 202],
                [4, 20, 252],
                [5, 4, 252],
                [4, 6, 252],
                [6, 19, 252],
                [7, 2, 152],
                [0, 8, 105],
                [1, 8, 95],
                [8, 5, 212],
                [4, 21, 202],
                [7, 8, 162],
                [5, 6, 252],
                [0, 3, 95],
                [8, 3, 162],
                [1, 7, 85],
            ],
            schema={"source_node": pl.Int32, "target_node": pl.Int32, "weight": pl.Int16},
        )
        assert_frame_equal(expected_edges, result_edges, check_row_order=False)

        hexagon_points, edges_linestrings = self.convert_nodes_and_edges_to_gdfs(
            hexagonal_grid, result_edges, hexagon_size=hexagon_size
        )
        assert all([length == pytest.approx(6.93, abs=0.01) for length in edges_linestrings.length])

        if debug:
            self.write_debug(hexagon_points, edges_linestrings, suffix="previous_row_only")

    def test_generate_edges_with_previous_blocks(
        self,
        hexagonal_grid: pl.DataFrame,
        block: pl.DataFrame,
        previous_edge_nodes_current_row: pl.DataFrame,
        relevant_edge_nodes_previous_row: pl.DataFrame,
        hexagon_size: int,
        debug: bool = False,
    ):
        """
        Test that verifies that all edges are present, have correct edge weights and length when generating edges
        for a block when having previous edges in both the current and previous row.

        This test can be understood most easily by setting debug=True and inspecting the results in QGis.
        """
        nodes_to_check = pl.concat([block, previous_edge_nodes_current_row, relevant_edge_nodes_previous_row])

        generator = HexagonEdgeGenerator()
        result_edges = generator.generate(block, nodes_to_check)

        expected_edges = pl.DataFrame(
            data=[
                [2, 20, 202],
                [3, 10, 202],
                [7, 5, 202],
                [2, 4, 202],
                [3, 6, 202],
                [3, 13, 152],
                [4, 19, 252],
                [2, 5, 202],
                [3, 5, 202],
                [4, 20, 252],
                [5, 4, 252],
                [4, 6, 252],
                [6, 19, 252],
                [7, 2, 152],
                [0, 8, 105],
                [1, 8, 95],
                [8, 5, 212],
                [4, 21, 202],
                [6, 10, 252],
                [6, 9, 252],
                [0, 13, 95],
                [7, 8, 162],
                [5, 6, 252],
                [0, 3, 95],
                [8, 3, 162],
                [1, 7, 85],
            ],
            schema={"source_node": pl.Int32, "target_node": pl.Int32, "weight": pl.Int16},
        )
        assert_frame_equal(expected_edges, result_edges, check_row_order=False)

        hexagon_points, edges_linestrings = self.convert_nodes_and_edges_to_gdfs(
            hexagonal_grid, result_edges, hexagon_size=hexagon_size
        )
        assert all([length == pytest.approx(6.93, abs=0.01) for length in edges_linestrings.length])

        if debug:
            self.write_debug(hexagon_points, edges_linestrings, suffix="previous_block_and_rows")

    @staticmethod
    def convert_nodes_and_edges_to_gdfs(
        grid: pl.DataFrame, edges: pl.DataFrame, hexagon_size: int
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        # Convert axial coordinates to cartesian coordinates
        q = grid["q"]
        r = grid["r"]
        x = q * (-3 / 2 * hexagon_size)
        y = (r + q / 2) * (math.sqrt(3) * hexagon_size)

        hexagon_points = gpd.GeoDataFrame(
            data=grid.select("node_id", "suitability_value"),
            geometry=gpd.points_from_xy(x, y),
            columns=["node_id", "suitability_value"],
            crs=Config.CRS,
        ).set_index("node_id")
        source_points = hexagon_points.loc[edges["source_node"], "geometry"]
        target_points = hexagon_points.loc[edges["target_node"], "geometry"]
        edges_linestrings = gpd.GeoDataFrame(
            data=edges["weight"],
            geometry=shapely.linestrings(
                np.stack([source_points.get_coordinates().values, target_points.get_coordinates().values], axis=1)
            ),
            columns=["weight"],
            crs=Config.CRS,
        )
        return hexagon_points, edges_linestrings

    @staticmethod
    def write_debug(hexagon_points: gpd.GeoDataFrame, edges_linestrings: gpd.GeoDataFrame, suffix: str):
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT,
            hexagon_points,
            f"pytest_hexagon_edges_test_nodes_{suffix}",
            overwrite=True,
        )
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT,
            edges_linestrings,
            f"pytest_hexagon_edges_test_nodes_{suffix}",
            overwrite=True,
        )
