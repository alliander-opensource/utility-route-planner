# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import geopandas as gpd
import polars as pl
import rustworkx as rx
import shapely
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_edge_generator import HexagonEdgeGenerator
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_grid_builder import (
    HexagonGridBuilder,
)
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import get_hexagon_width_and_height
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
        self.hexagon_width, self.hexagon_height = get_hexagon_width_and_height(hexagon_size)

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
        is_edge: list[bool] = []

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

            block_edge_attributes = block.select("node_id", "suitability_value", "q", "r")
            blocks_to_check = pl.concat([previous_row, current_row, block_edge_attributes])
            for edges in hexagon_edge_generator.generate(block, blocks_to_check):
                self.graph.add_edges_from(edges)

            # Store the edges of the current block for edge generation in the next block. In case this was the final
            # block of this row, the previous row is set to this row and current_row is reset to the last block.
            edge_coordinates = self.get_block_edge_coordinates(block)
            is_edge.extend(block["node_id"].is_in(edge_coordinates["node_id"]))

            if not final_column:
                current_row = pl.concat([current_row, edge_coordinates])
            else:
                previous_row = current_row
                current_row = block_edge_attributes

        nodes_gdf = gpd.GeoDataFrame(
            data={"node_id": node_ids, "suitability_value": node_suitability_values, "is_edge": is_edge},
            geometry=gpd.points_from_xy(x=node_x_coordinates, y=node_y_coordinates, crs=Config.CRS),
        )

        return self.graph, nodes_gdf

    def get_block_edge_coordinates(self, block_coordinates: pl.DataFrame) -> pl.DataFrame:
        """
        Given the coordinates of a block, get left side and bottom coordinates
        """
        min_x_coordinate = block_coordinates["x"].min()
        min_y_coordinate = block_coordinates["y"].min()

        edge_coordinates = block_coordinates.filter(
            (pl.col("x") == min_x_coordinate) | ((pl.col("y") - min_y_coordinate) <= self.hexagon_height)
        )

        return edge_coordinates.select("node_id", "suitability_value", "q", "r")
