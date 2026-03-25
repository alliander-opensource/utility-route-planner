# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import geopandas as gpd
import polars as pl
import rustworkx as rx
import shapely
import structlog

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

        # node_ids: list[int] = []
        # node_x_coordinates: list[float] = []
        # node_y_coordinates: list[float] = []

        current_row_edge_coordinates = pl.DataFrame(
            schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32}
        )
        previous_row_edge_coordinates = pl.DataFrame(
            schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32}
        )
        previous_block_edge_coordinates = pl.DataFrame(
            schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32}
        )

        for block, last_column in grid_constructor.construct_grid(self.project_area):
            block_node_ids = self.graph.add_nodes_from(block.select("suitability_value", "x", "y").rows())
            block = block.with_columns(pl.Series("node_id", list(block_node_ids), dtype=pl.Int32))

            # Store all block information. Create a temporary dict to store all information of the block for edge
            # processing.
            # node_ids.extend(block_node_ids)
            # node_x_coordinates.extend(block["x"])
            # node_y_coordinates.extend(block["y"])

            block_edge_attributes = block.select("node_id", "suitability_value", "q", "r")
            block_edge_coordinates = self.get_block_edge_coordinates(block)
            current_row_edge_coordinates = pl.concat([current_row_edge_coordinates, block_edge_coordinates])

            relevant_previous_row_nodes = previous_row_edge_coordinates.filter(
                pl.col("q").is_between(block_edge_attributes["q"].min() - 1, block_edge_attributes["q"].max() + 1)
            )
            nodes_to_check = pl.concat(
                [block_edge_attributes, previous_block_edge_coordinates, relevant_previous_row_nodes]
            )
            edges = hexagon_edge_generator.generate(block_edge_attributes, nodes_to_check)
            self.graph.add_edges_from(edges)

            if last_column:
                previous_row_edge_coordinates = current_row_edge_coordinates
                current_row_edge_coordinates = pl.DataFrame(
                    schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32}
                )
                previous_block_edge_coordinates = pl.DataFrame(
                    schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32}
                )
            else:
                previous_block_edge_coordinates = block_edge_coordinates

        # nodes_gdf = gpd.GeoDataFrame(
        #     data={"node_id": node_ids},
        #     geometry=gpd.points_from_xy(x=node_x_coordinates, y=node_y_coordinates, crs=Config.CRS),
        # )

        return self.graph, gpd.GeoDataFrame()

    def get_block_edge_coordinates(self, block_coordinates: pl.DataFrame) -> pl.DataFrame:
        """
        Given the coordinates of a block, get right side and bottom coordinates
        """
        min_x_coordinate = block_coordinates["x"].min()
        min_y_coordinate = block_coordinates["y"].min()

        edge_coordinates = block_coordinates.filter(
            (pl.col("x") == min_x_coordinate) | (abs(pl.col("y") - min_y_coordinate) <= 0.6 * self.hexagon_height)
        ).select("node_id", "suitability_value", "q", "r")

        return edge_coordinates
