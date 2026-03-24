#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import rustworkx as rx
import shapely
import structlog
from geopandas import GeoDataFrame

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import HexagonEdgeInfo, PipeRammingEdgeInfo
from utility_route_planner.util.geo_utilities import get_empty_geodataframe
from utility_route_planner.util.timer import time_function

logger = structlog.get_logger(__name__)


def get_hexagon_width_and_height(hexagon_size: float) -> tuple[float, float]:
    """
    Compute hexagon width and height, given the provided size of the hexagon. In this calculation, width and height are
    computed for a flat-top oriented hexagon.

    source: https://www.redblobgames.com/grids/hexagons/#basics

    :param hexagon_size: size of hexagon described by the inner circle of the hexagon that touches the edges
    :return: tuple consisting of two floats that represent the width and height of the hexagon
    """

    hexagon_width = 2 * hexagon_size
    hexagon_height = math.sqrt(3) * hexagon_size

    return hexagon_width, hexagon_height


@time_function
def convert_hexagon_graph_to_gdfs(
    hexagon_graph: rx.PyGraph, edges: bool = True
) -> tuple[GeoDataFrame, None] | GeoDataFrame:
    if hexagon_graph.num_nodes() == 0:
        logger.warning("Hexagon graph is empty, returning empty GeoDataFrame.")
        return get_empty_geodataframe()

    nodes_gdf = gpd.GeoDataFrame(hexagon_graph.nodes(), crs=Config.CRS)

    if edges:
        edge_keys = pd.DataFrame(hexagon_graph.edge_list(), columns=["u", "v"])
        edge_attributes = gpd.GeoDataFrame(hexagon_graph.edges())
        edges_gdf = gpd.GeoDataFrame(pd.concat([edge_keys, edge_attributes], axis=1), crs=Config.CRS)
        u_coords = nodes_gdf.loc[edges_gdf["u"]].get_coordinates().values
        v_coords = nodes_gdf.loc[edges_gdf["v"]].get_coordinates().values

        # Stack u and v coordinates on axis 1 to get correct linestring coordinate format: [[u_x, u_y], [v_x, v_y]]
        line_string_coords = np.stack([u_coords, v_coords], axis=1)
        edge_line_strings = shapely.linestrings(line_string_coords)

        edges_gdf = edges_gdf.set_geometry(edge_line_strings, crs=Config.CRS)
        return nodes_gdf, edges_gdf
    else:
        return nodes_gdf


def get_hexagon_edge_weight(hexagon_edge: float | HexagonEdgeInfo) -> float:
    """
    When constructing the Hexagon graph, an edge can be set in two ways:
    - When set in the HexagonGraphBuilder: weight is set as a float directly
    - When set as part of piperamming: weight is set as part of the HexagonEdgeInfo dataclass

    This function can be used to extract the edge weight when doing a shortest path analysis.
    """
    if isinstance(hexagon_edge, float):
        return hexagon_edge
    else:
        return hexagon_edge.weight


def get_hexagon_edge_geometries_for_path(
    graph: rx.PyGraph, path_node_indices: list[int], nodes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    edges_list = []
    for start_node, end_node in zip(path_node_indices, path_node_indices[1:]):
        edge_data = graph.get_edge_data(start_node, end_node)
        edge_linestring = shapely.LineString([nodes.loc[start_node, "geometry"], nodes.loc[end_node, "geometry"]])
        if isinstance(edge_data, PipeRammingEdgeInfo):
            edge_meta_data = asdict(edge_data)
        else:
            edge_meta_data = dict(weight=get_hexagon_edge_weight(edge_data), geometry=edge_linestring)

        edges_list.append(edge_meta_data)
    return gpd.GeoDataFrame(data=edges_list, crs=Config.CRS)


def convert_hexagon_edges_to_gdf(graph: rx.PyGraph, nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert all edges in a Hexagon graph to a GeoDataframe.

    Note: when supplying large graphs, this function can take some time to complete due to the geometry creation
    of all edges in the graph.

    :param graph: graph to convert edges for
    :param nodes: all nodes in the graph as a geodataframe containing the source and target geometries
    :return: geodataframe with all edges, edge weights and geometries from the input graph
    """
    node_to_geom_mapping = nodes.set_index("node_id")["geometry"]

    edge_weight_map = graph.edge_index_map()
    source_nodes = [source_node for source_node, _, _ in edge_weight_map.values()]
    target_nodes = [target_node for _, target_node, _ in edge_weight_map.values()]
    weights = [get_hexagon_edge_weight(weight) for _, _, weight in edge_weight_map.values()]

    source_coordinates = node_to_geom_mapping.loc[source_nodes].get_coordinates().values
    target_coordinates = node_to_geom_mapping.loc[target_nodes].get_coordinates().values
    edge_geometries = shapely.linestrings(np.stack([source_coordinates, target_coordinates], axis=1))

    return gpd.GeoDataFrame(
        {"source_node": source_nodes, "target_node": target_nodes, "weight": weights, "geometry": edge_geometries},
        crs=Config.CRS,
    )
