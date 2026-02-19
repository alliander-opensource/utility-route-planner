# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import geopandas as gpd
import rustworkx as rx
import shapely
import structlog

from utility_route_planner.models.multilayer_network.graph_datastructures import TempNode
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
    def build_graph(self) -> rx.PyGraph:
        grid_constructor = HexagonGridBuilder(
            self.raster_groups, self.preprocessed_vectors, self.hexagon_size, self.block_size
        )

        hexagon_edge_generator = HexagonEdgeGenerator()
        previous_row: dict[tuple[int, int], TempNode] = {}
        current_row: dict[tuple[int, int], TempNode] = {}
        for block, final_column in grid_constructor.construct_grid(self.project_area):
            suitability_values = block.loc[:, "suitability_value"].values
            node_ids = self.graph.add_nodes_from(suitability_values)
            block.index = node_ids

            block_coordinates: dict[tuple[int, int], TempNode] = {
                (node.axial_q, node.axial_r): TempNode(node.Index, node.suitability_value)
                for node in block.itertuples()
            }

            if not final_column:
                current_row.update(block_coordinates)
            else:
                previous_row = current_row
                current_row = block_coordinates

            blocks_to_check = previous_row | current_row
            for edges in hexagon_edge_generator.generate(block, blocks_to_check):
                self.graph.add_edges_from(edges)

        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        logger.info(
            f"Max degree: {max(degrees)}, Min degree: {min(degrees)}, avg: {sum(degrees) / self.graph.num_nodes()}"
        )
        logger.info(
            f"Graph has {self.graph.num_nodes()} nodes & {self.graph.num_edges()} edges for hexagon_size {self.hexagon_size}"
        )

        return self.graph
