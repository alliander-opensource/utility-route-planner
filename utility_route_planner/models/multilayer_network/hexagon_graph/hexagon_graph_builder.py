# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict

import geopandas as gpd
import pandas as pd
import polars as pl
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
        osm_edges: gpd.GeoDataFrame,
        hexagon_size: float,
        block_size: int,
    ):
        self.project_area = project_area
        self.raster_groups = raster_groups
        self.preprocessed_vectors = preprocessed_vectors
        self.osm_edges = osm_edges
        self.hexagon_size = hexagon_size
        self.block_size = block_size
        self.graph = rx.PyGraph()

    @time_function
    def build_graph(self) -> rx.PyGraph:
        grid_constructor = HexagonGridBuilder(
            self.raster_groups, self.preprocessed_vectors, self.osm_edges, self.hexagon_size, self.block_size
        )

        hexagon_edge_generator = HexagonEdgeGenerator()
        previous_row: dict[tuple[int, int], TempNode] = {}
        current_row: dict[tuple[int, int], TempNode] = {}

        node_ids: list[int] = []
        node_suitability_values: list[int] = []
        node_x_coordinates: list[float] = []
        node_y_coordinates: list[float] = []
        node_near_osm_edge: list[bool] = []

        for block, final_column in grid_constructor.construct_grid(self.project_area):
            suitability_values = block.loc[:, "suitability_value"].values
            block_node_ids = self.graph.add_nodes_from(suitability_values)
            block.index = block_node_ids

            block_coordinates: dict[tuple[int, int], TempNode] = {}
            for node in block.itertuples():
                node_ids.append(node.Index)
                node_suitability_values.append(node.suitability_value)
                node_x_coordinates.append(node.x)
                node_y_coordinates.append(node.y)
                node_near_osm_edge.append(node.near_osm_edge)
                block_coordinates[(node.axial_q, node.axial_r)] = TempNode(node.Index, node.suitability_value)

            blocks_to_check = previous_row | current_row | block_coordinates
            blocks_to_check_df = pd.DataFrame(
                [{"axial_q": q, "axial_r": r, **asdict(v)} for (q, r), v in blocks_to_check.items()]
            )
            blocks_to_check_df = blocks_to_check_df.set_index(keys=["node_id"])

            for edges in hexagon_edge_generator.generate(block, blocks_to_check_df):
                self.graph.add_edges_from(edges)

            # Max r represents the bottom of the block
            max_r = max(r for _, r in block_coordinates)

            # Max q represents the bottom of the block
            max_q = max(q for q, _ in block_coordinates)

            # Get all coordinates on the edges of the blocks
            edge_coordinates = {
                (q, r): value for (q, r), value in block_coordinates.items() if q == max_q or r == max_r
            }

            if not final_column:
                current_row.update(edge_coordinates)
            else:
                previous_row = current_row
                current_row = edge_coordinates

        nodes_df = pl.DataFrame(
            {
                "node_id": node_ids,
                "suitability_value": node_suitability_values,
                "x": node_x_coordinates,
                "y": node_y_coordinates,
                "near_osm_edge": node_near_osm_edge,
            },
            schema={
                "node_id": pl.Int32,
                "suitability_value": pl.Float32,
                "x": pl.Float64,
                "y": pl.Float64,
                "near_osm_edge": pl.Boolean,
            },
        )
        logger.info(f"Nodes df estimated size: {nodes_df.estimated_size(unit='gb')}gb")

        return self.graph
