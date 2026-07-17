# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import geopandas as gpd
import numpy as np
import polars as pl
import rustworkx as rx
import shapely
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import hexagon_edge_info, HexagonNodeInfo
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

    def __init__(self, grid_builder: HexagonGridBuilder, edge_generator: HexagonEdgeGenerator):
        self.hexagon_size = grid_builder.hexagon_size
        self.hexagon_width, self.hexagon_height = get_hexagon_width_and_height(self.hexagon_size)
        self.grid_builder = grid_builder
        self.edge_generator = edge_generator

    @time_function
    def build_graph(
        self,
        project_area: shapely.Polygon | shapely.MultiPolygon,
        raster_groups: dict[str, str],
        preprocessed_vectors: dict[str, gpd.GeoDataFrame],
    ) -> tuple[rx.PyGraph, gpd.GeoDataFrame]:
        # Left-side and bottom coordinates for all blocks in the current row.
        current_row_edge_coordinates = pl.DataFrame(
            schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32}
        )

        # Left-side and bottom coordinates for all blocks in the previous row. When finishing a row, this
        # is set by the current_row_edge_coordinates dataframe. It is used to connect the top side of blocks
        # in the current row to the previous row.
        previous_row_edge_coordinates = self._get_empty_nodes_df()

        # Edge coordinates of the previous block in the current row. It is used to create edges from the current to
        # the previous block in the current row.
        previous_block_edge_coordinates = self._get_empty_nodes_df()

        x_matrix, y_matrix = self.grid_builder.construct_hexagonal_grid_for_bounding_box(project_area)

        # Before running, initialize a numpy structured array for storing all node data while constructing the graph. As
        # it is based on the bounding box, -1 is used as fill value such that it can be filtered later on. Using the
        # pre-initialized array is much more efficient than concatenating all node data to a list while constructing
        # the graph.
        n_nodes = x_matrix.shape[0] * x_matrix.shape[1]
        nodes = np.full(
            n_nodes,
            fill_value=-1,
            dtype=[("node_id", np.int32), ("suitability_value", np.int16), ("x", np.float32), ("y", np.float32)],
        )

        # Construct hexagonal graph using a sliding-window approach
        graph = rx.PyGraph()
        for block, last_column in self.grid_builder.construct_grid_blocks(
            x_matrix, y_matrix, preprocessed_vectors, raster_groups
        ):
            # An empty block still carries the row/column position, so we must run the boundary bookkeeping below even
            # though there are no nodes or edges to add. Skipping it entirely would drop the final-column signal and
            # leave the next row unable to connect upward, fragmenting the graph into disconnected bands.
            if block is not None:
                graph, block_node_ids = self._add_nodes_to_graph(graph, block)

                # Add node id to the block dataframe for edge processing
                block = block.with_columns(pl.Series("node_id", list(block_node_ids), dtype=pl.Int32))

                # Store all block information in the total node array
                nodes["node_id"][block_node_ids] = block_node_ids
                nodes["suitability_value"][block_node_ids] = block["suitability_value"]
                nodes["x"][block_node_ids] = block["x"]
                nodes["y"][block_node_ids] = block["y"]

                # Select all attributes in the current block which are relevant for edge generation
                block_edge_attributes = block.select("node_id", "suitability_value", "q", "r")
                block_edge_coordinates = self.get_block_edge_coordinates(block)
                current_row_edge_coordinates = pl.concat([current_row_edge_coordinates, block_edge_coordinates])

                # Only check previous row nodes that are on top of the current block to reduce unnecessary joins.
                # Filtering is performed using the min-max q-values + a buffer of 1 to include the boundaries as well.
                relevant_previous_row_nodes = previous_row_edge_coordinates.filter(
                    pl.col("q").is_between(block_edge_attributes["q"].min() - 1, block_edge_attributes["q"].max() + 1)
                )

                # For edge determination, the edge generator must consider nodes within the block, boundary nodes from
                # the previous block and boundary nodes from the previous row to make sure the graph is connected.
                nodes_to_check = pl.concat(
                    [block_edge_attributes, previous_block_edge_coordinates, relevant_previous_row_nodes]
                )

                graph = self._add_edges_to_graph(graph, block_edge_attributes, nodes_to_check)

                # Non-empty block: this block becomes the previous block for the next block in the row. When the row
                # ends (last_column), this is overwritten by the reset below.
                next_previous_block_edge_coordinates = block_edge_coordinates
            else:
                # An empty block represents a gap in the row. Reset the previous-block reference so the next non-empty
                # block is not incorrectly bridged across the gap.
                next_previous_block_edge_coordinates = self._get_empty_nodes_df()

            if last_column:
                # If the final column of the current row is reached, set all edge coordinates of the current row as
                # previous row edge coordinates. This way, they can be used to determine connecting upper edges in the
                # next row. All other dataframes containing previous coordinates are reset. This must run even for an
                # empty final-column block, otherwise the next row cannot connect to the current row.
                previous_row_edge_coordinates = current_row_edge_coordinates
                current_row_edge_coordinates = self._get_empty_nodes_df()
                previous_block_edge_coordinates = self._get_empty_nodes_df()
            else:
                # If the final column of the current row is not yet reached, set the previous block to the current
                # block. This way, it can be connected to the next block in the current row.
                previous_block_edge_coordinates = next_previous_block_edge_coordinates

        logger.info(f"Number of connected components: {len(rx.connected_components(graph))}")
        nodes_gdf = self._convert_nodes_to_gdf(nodes)
        return graph, nodes_gdf

    @staticmethod
    def _add_nodes_to_graph(graph: rx.PyGraph, block: pl.DataFrame) -> tuple[rx.PyGraph, rx.NodeIndices]:
        """
        Add nodes to the graph and set node id on the node payload.
        """
        node_payloads = [HexagonNodeInfo(weight=weight) for weight in block["suitability_value"].to_list()]
        block_node_ids = graph.add_nodes_from(node_payloads)

        # Assign node id to node payloads and block dataframe
        [block_data_object.set_node_id(node_id) for block_data_object, node_id in zip(node_payloads, block_node_ids)]

        return graph, block_node_ids

    def _add_edges_to_graph(
        self, graph: rx.PyGraph, block_edge_attributes: pl.DataFrame, nodes_to_check: pl.DataFrame
    ) -> rx.PyGraph:
        """
        Add edges to the graph and set edge id on the edge payload.
        """
        edges = self.edge_generator.generate(block_edge_attributes, nodes_to_check)
        edge_ids = graph.add_edges_from(edges.rows())
        for edge_id, (u, v, weight) in zip(edge_ids, edges.rows()):
            graph.update_edge(u, v, hexagon_edge_info(edge_id, weight))
        return graph

    def get_block_edge_coordinates(self, block_coordinates: pl.DataFrame) -> pl.DataFrame:
        """
        Given the coordinates of a block, get left side and bottom coordinates.
        """
        min_x_coordinate = block_coordinates["x"].min()
        min_y_coordinate = block_coordinates["y"].min()

        edge_coordinates = block_coordinates.filter(
            (pl.col("x") == min_x_coordinate) | (abs(pl.col("y") - min_y_coordinate) <= 0.6 * self.hexagon_height)
        ).select("node_id", "suitability_value", "q", "r")

        return edge_coordinates

    @staticmethod
    def _get_empty_nodes_df() -> pl.DataFrame:
        return pl.DataFrame(schema={"node_id": pl.Int32, "suitability_value": pl.Int16, "q": pl.Int32, "r": pl.Int32})

    @staticmethod
    def _convert_nodes_to_gdf(nodes: np.ndarray) -> gpd.GeoDataFrame:
        """
        Only include filled rows for node geodataframe conversion (placeholder rows can be identified with node_id==-1)
        """
        nodes = np.extract(nodes["node_id"] >= 0, nodes)
        node_gdf = gpd.GeoDataFrame(
            data={"node_id": nodes["node_id"], "suitability_value": nodes["suitability_value"]},
            geometry=gpd.points_from_xy(x=nodes["x"], y=nodes["y"], crs=Config.CRS),
        )
        node_gdf = node_gdf.set_index("node_id")
        return node_gdf
