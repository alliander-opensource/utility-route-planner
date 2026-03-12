# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import geopandas as gpd
import pandas as pd
import polars as pl
import rustworkx as rx
import shapely
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_edge_generator import HexagonEdgeGenerator
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_grid_builder import (
    HexagonGridBuilder,
)
from utility_route_planner.util.timer import time_function

logger = structlog.get_logger(__name__)


class HexagonGraphBuilder:
    """
    Class is used to construct a spatial graph in flat-top hexagonal structure given a set of spatial input
    vectors. Each node and edge have an assigned suitability value that is computed based on the location
    and intersecting vector.
    """

    def __init__(
        self,
        project_area: shapely.Polygon,
        raster_groups: dict[str, str],
        preprocessed_vectors: dict[str, gpd.GeoDataFrame],
        hexagon_size: float,
        block_size: int,
    ):
        self.project_area = project_area
        self.raster_groups = raster_groups
        self.preprocessed_vectors = preprocessed_vectors
        self.hexagon_size = hexagon_size
        self.block_size = block_size
        self.graph = rx.PyGraph()

    @time_function
    def build_graph(self) -> tuple[rx.PyGraph, gpd.GeoDataFrame]:
        grid_constructor = HexagonGridBuilder(
            self.raster_groups, self.preprocessed_vectors, self.hexagon_size, self.block_size
        )
        hexagon_edge_generator = HexagonEdgeGenerator()

        previous_row: pl.DataFrame = pl.DataFrame(
            schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32}
        )
        current_row: pl.DataFrame = pl.DataFrame(
            schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32}
        )

        node_ids: list[int] = []
        node_suitability_values: list[int] = []
        node_x_coordinates: list[float] = []
        node_y_coordinates: list[float] = []

        for i, (block, final_column) in enumerate(grid_constructor.construct_grid(self.project_area)):
            suitability_values = block["suitability_value"]
            block_node_ids = self.graph.add_nodes_from(suitability_values)
            block = block.with_columns(pl.Series("node_id", list(block_node_ids), dtype=pl.Int32))

            # Store all block information. Create a temporary dict to store all information of the block for edge
            # processing.
            node_ids.extend(block_node_ids)
            node_suitability_values.extend(suitability_values)
            node_x_coordinates.extend(block["x"])
            node_y_coordinates.extend(block["y"])

            blocks_to_check = pl.concat(
                [previous_row, current_row, block.select("node_id", "suitability_value", "q", "r")]
            )
            for edges in hexagon_edge_generator.generate(block, blocks_to_check):
                self.graph.add_edges_from(edges)

            # Store the edges of the current block for edge generation in the next block. In case this was the final
            # block of this row, the previous row is set to this row and current_row is reset to the last block.
            edge_coordinates = self.get_block_edge_coordinates(block)

            if not final_column:
                current_row = pd.concat([current_row, edge_coordinates])
            else:
                previous_row = current_row
                current_row = edge_coordinates

        nodes_gdf = gpd.GeoDataFrame(
            data={
                "node_id": node_ids,
                "suitability_value": node_suitability_values,
            },
            geometry=gpd.points_from_xy(x=node_x_coordinates, y=node_y_coordinates, crs=Config.CRS),
        )

        return self.graph, nodes_gdf

    @staticmethod
    def get_block_edge_coordinates(block_coordinates: pd.DataFrame) -> pd.DataFrame:
        """
        Given the coordinates of a block, get left side and bottom coordinates. The left edge are equal to the max
        axial q (due to the coordinate project being used). The bottom coordinates can be found by solving the equation
        (Δq, Δr) = (-2, +1), which means that q-2, r+1 if we do one step to the right on the grid. The solution to this
        equation is q + 2r. By checking which coordinates are equal to this formulation, we can find the bottom corner
        of the hexagon block.

        :param block_coordinates: axial coordinates, node ids and suitability values for all nodes within a block
        :return: edge coordinates of the block including corresponding node ids and suitability values.
        """
        block_coordinates_cp = block_coordinates.copy()

        # Max q represents the horizontal (left) edge of the block
        max_q = block_coordinates_cp["axial_q"].max()

        # In this coordinate system, the bottom edge follows a diagonal where q + 2r is minimal.
        # Include one extra row (min_diagonal + 2) to account for the hex offset between columns.

        # Horizontal coordinates follow rule : q + 2r. By checking where this equation is minimal in the block, we can
        # find the bottom edge.
        bottom_coordinate_reference = (block_coordinates_cp["axial_q"] + 2 * block_coordinates_cp["axial_r"]).min()

        edge_coordinates = block_coordinates_cp.loc[
            (block_coordinates_cp["axial_q"] == max_q)
            | (block_coordinates_cp["axial_q"] + 2 * block_coordinates_cp["axial_r"] == bottom_coordinate_reference)
        ]
        return edge_coordinates
