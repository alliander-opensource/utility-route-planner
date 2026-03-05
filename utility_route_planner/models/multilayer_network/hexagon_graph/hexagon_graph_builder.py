# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import geopandas as gpd
import pandas as pd
import rustworkx as rx
import shapely
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import TempNode
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_grid_builder import (
    HexagonGridBuilder,
)
from utility_route_planner.util.timer import time_function
from utility_route_planner.util.write import write_results_to_geopackage

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

        # hexagon_edge_generator = HexagonEdgeGenerator()
        previous_row: dict[tuple[int, int], TempNode] = {}
        current_row: dict[tuple[int, int], TempNode] = {}

        node_ids: list[int] = []
        node_suitability_values: list[int] = []
        node_x_coordinates: list[float] = []
        node_y_coordinates: list[float] = []

        for i, (block, final_column) in enumerate(grid_constructor.construct_grid(self.project_area)):
            suitability_values = block.loc[:, "suitability_value"].values
            block_node_ids = self.graph.add_nodes_from(suitability_values)
            block.index = block_node_ids

            # block_geoms = gpd.points_from_xy(block["x"], block["y"], crs=Config.CRS)
            # block_gdf = gpd.GeoDataFrame(block, geometry=block_geoms)
            #
            # write_results_to_geopackage(Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, block_gdf, f"block_nodes_{i}", overwrite=True)

            # Store all block information. Create a temporary dict to store all information of the block for edge
            # processing.
            block_coordinates: dict[tuple[int, int], TempNode] = {}
            for node in block.itertuples():
                node_ids.append(node.Index)
                node_suitability_values.append(node.suitability_value)
                node_x_coordinates.append(node.x)
                node_y_coordinates.append(node.y)
                block_coordinates[(node.axial_q, node.axial_r)] = node.Index

            # Determine which previous block must be included into the edge generation to reduce the number of neighbour
            # candidate calls in the edge generation.
            previous_blocks_to_check = self.filter_previous_blocks(block_coordinates, previous_row, current_row)
            blocks_to_check = previous_blocks_to_check | block_coordinates
            blocks_to_check_df = pd.DataFrame(
                [{"axial_q": q, "axial_r": r, "node_id": v} for (q, r), v in blocks_to_check.items()]
            )
            blocks_to_check_df = blocks_to_check_df.set_index(keys=["node_id"])

            # for edges in hexagon_edge_generator.generate(block, blocks_to_check_df):
            #     self.graph.add_edges_from(edges)

            # Store the edges of the current block for edge generation in the next block. In case this was the final
            # block of this row, the previous row is set to this row and current_row is reset to the last block.
            edge_coordinates = self.get_block_edge_coordinates(block_coordinates)
            block_select = block.loc[edge_coordinates.values()]

            block_edge_geoms = gpd.points_from_xy(block_select["x"], block_select["y"], crs=Config.CRS)
            block_edge_gdf = gpd.GeoDataFrame(block_select, geometry=block_edge_geoms)

            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, block_edge_gdf, f"block_edge_nodes_{i}", overwrite=True
            )

            if not final_column:
                current_row.update(edge_coordinates)
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
    def filter_previous_blocks(
        block_coordinates: dict[tuple[int, int], TempNode],
        previous_row: dict[tuple[int, int], TempNode],
        current_row: dict[tuple[int, int], TempNode],
    ):
        """
        Only check blocks that could be adjacent to the current block. This is the case if it is exactly 1-q or 1-r away
        from the top or left side of the block

        :param block_coordinates: coordinates of current block
        :param previous_row: all coordinates of the previous row
        :param current_row: all coordinates of the current row so far

        :return: previous blocks which are adjacent to the current block
        """
        min_q = min(q for q, _ in block_coordinates)
        min_r = min(r for _, r in block_coordinates)

        previous_blocks_to_check = {
            (q, r): value for (q, r), value in (previous_row | current_row).items() if q >= min_q - 1 and r >= min_r - 1
        }
        return previous_blocks_to_check

    @staticmethod
    def get_block_edge_coordinates(
        block_coordinates: dict[tuple[int, int], TempNode],
    ) -> dict[tuple[int, int], TempNode]:
        """
        Given the coordinates of a block, get left side and bottom coordinates. The left edge are equal to the max
        axial q (due to the coordinate project being used). The bottom coordinates can be found by solving the equation
        (Δq, Δr) = (-2, +1), which means that q-2, r+1 if we do one step to the right on the grid. The solution to this
        equation is q + 2r. By checking which coordinates are equal to this formulation, we can find the bottom corner
        of the hexagon block.

        :param block_coordinates: axial coordinates, node ids and suitability values for all nodes within a block
        :return: edge coordinates of the block including corresponding node ids and suitability values.
        """
        # Max q represents the horizontal (left) edge of the block
        max_q = max(q for q, _ in block_coordinates)

        # In this coordinate system, the bottom edge follows a diagonal where q + 2r is minimal.
        # Include one extra row (min_diagonal + 2) to account for the hex offset between columns.

        # Horizontal coordinates follow rule : q + 2r. By checking where this equation is minimal in the block, we can
        # find the bottom edge.
        bottom_coordinate_reference = min(q + 2 * r for q, r in block_coordinates)

        edge_coordinates = {
            (q, r): value
            for (q, r), value in block_coordinates.items()
            if q == max_q or q + 2 * r == bottom_coordinate_reference
        }
        return edge_coordinates
