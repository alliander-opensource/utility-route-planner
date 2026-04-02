#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0
import geopandas as gpd
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest
import shapely

from settings import Config
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_edge_generator import HexagonEdgeGenerator
from utility_route_planner.util.write import reset_geopackage, write_results_to_geopackage


class TestHexagonEdgeGenerator:
    @pytest.fixture()
    def hexagonal_grid(self) -> pl.DataFrame:
        """
        Hexagonal grid generated with hexagonsize=4

        """
        return pl.DataFrame(
            data=[
                [0, -29159, 79677, 126, 174954.55, 451007.03],
                [1, -29160, 79678, 76, 174960.55, 451010.50],
                [2, -29161, 79678, 9, 174966.55, 451007.03],
                [3, -29160, 79677, 76, 174960.55, 451003.56],
                [4, -29159, 79676, 76, 174954.55, 451000.09],
                [5, -29160, 79676, 86, 174960.55, 450996.63],
                [6, -29161, 79677, 9, 174966.55, 451000.09],
            ],
            schema={
                "node_id": pl.Int32,
                "q": pl.Int32,
                "r": pl.Int32,
                "suitability_value": pl.Int16,
                "x": pl.Float32,
                "y": pl.Float32,
            },
        )

    def test_generate_edges_without_previous_blocks(self, hexagonal_grid: pl.DataFrame, debug: bool = False):
        """
        Test that verifies that all edges are present, have correct edge weights and length when generating edges
        for a block without having previous edges. This means only edges within the block itself are expected.

        This test can be understood most easily by setting debug=True and inspecting the results in QGis.
        """
        generator = HexagonEdgeGenerator()

        empty_previous_edges = pl.DataFrame(
            schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32}
        )
        nodes_to_check = pl.concat(
            [hexagonal_grid.select("node_id", "suitability_value", "q", "r"), empty_previous_edges]
        )
        result_edges = generator.generate(hexagonal_grid, nodes_to_check)

        expected_edges = pl.DataFrame(
            data=[
                [5, 6, 95],
                [3, 6, 85],
                [4, 0, 202],
                [0, 3, 202],
                [3, 2, 85],
                [4, 5, 162],
                [3, 1, 152],
                [1, 2, 85],
                [6, 2, 18],
                [5, 3, 162],
                [4, 3, 152],
                [0, 1, 202],
            ],
            schema={"source_node": pl.Int32, "target_node": pl.Int32, "weight": pl.Int16},
        )
        assert_frame_equal(expected_edges, result_edges, check_row_order=False)

        hexagon_points, edges_linestrings = self.convert_nodes_and_edges_to_gdfs(hexagonal_grid, result_edges)
        assert all([length == pytest.approx(6.93, abs=0.01) for length in edges_linestrings.length])

        if debug:
            reset_geopackage(Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT)

            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT,
                hexagon_points,
                "pytest_graph_nodes_no_previous_blocks",
                overwrite=True,
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT,
                edges_linestrings,
                "pytest_graph_edges_no_previous_blocks",
                overwrite=True,
            )

    @staticmethod
    def convert_nodes_and_edges_to_gdfs(
        grid: pl.DataFrame, edges: pl.DataFrame
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        hexagon_points = gpd.GeoDataFrame(
            data=grid.select("node_id", "suitability_value"),
            geometry=gpd.points_from_xy(grid["x"], grid["y"]),
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
