#  SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#  #
#  SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
import math

import geopandas as gpd
import numpy as np
import rustworkx as rx
import shapely
import structlog

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import (
    HexagonConnectionEdgeInfo,
    BaseWeightedEdgeInfo,
    hexagon_edge_info,
)

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


def get_inradius(hexagon_size: float) -> float:
    """Get the inradius of a hexagon."""
    return math.sqrt(3) * hexagon_size / 2


def build_flat_top_hexagons(hexagon_centroid_grid: gpd.GeoDataFrame, hexagon_size: float) -> np.ndarray:
    """
    Vectorized construction of flat-top hexagon polygons from their centroids.

    :param hexagon_centroid_grid: GeoDataFrame containing the centroids of the hexagons
    :param hexagon_size: inner circle of the hexagon
    :return: numpy array of shapely.Polygon objects, one per input centroid
    """
    coords = hexagon_centroid_grid.get_coordinates()
    x, y = coords["x"].to_numpy(), coords["y"].to_numpy()

    inradius = get_inradius(hexagon_size)

    # Vertex offsets for a flat-top hexagon, closed ring (first vertex repeated at the end).
    dx = np.array(
        [
            hexagon_size,
            hexagon_size / 2,
            -hexagon_size / 2,
            -hexagon_size,
            -hexagon_size / 2,
            hexagon_size / 2,
            hexagon_size,
        ]
    )
    dy = np.array([0.0, inradius, inradius, 0.0, -inradius, -inradius, 0.0])

    # Broadcast centroids against the constant offsets to a (N, 7, 2) coordinate array.
    coords = np.empty((x.size, dx.size, 2), dtype=np.float64)
    coords[:, :, 0] = x[:, None] + dx[None, :]
    coords[:, :, 1] = y[:, None] + dy[None, :]

    return shapely.polygons(coords)


def update_edge_id(
    new_id: int, hexagon_edge: hexagon_edge_info | BaseWeightedEdgeInfo
) -> hexagon_edge_info | BaseWeightedEdgeInfo:
    """
    Set the edge id as property of the edge, based on the type of edge that is encountered
    - When set in the HexagonGraphBuilder: id is set as in as part of the edge data
    - When set as part of piperamming or graph composing: id is set as part of the BaseWeightedEdgeInfo dataclass
    """
    match hexagon_edge:
        case hexagon_edge_info():
            return hexagon_edge_info(new_id, hexagon_edge[1])
        case BaseWeightedEdgeInfo():
            hexagon_edge.set_edge_id(new_id)
            return hexagon_edge
        case _:
            raise ValueError("Encountered invalid edge type")


def get_hexagon_node_geometry(nodes: gpd.GeoDataFrame, node_id: int) -> shapely.Point:
    return nodes.loc[node_id].geometry


def get_hexagon_edge_geometries_for_path(
    graph: rx.PyGraph, path_node_indices: list[int], hexagon_nodes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Given a list of node indices which represent a path on the graph, construct a dataframe to represent the path as a
    list of linestrings (edges).

    :param graph: graph for which the path was calculated
    :param path_node_indices: list of node indices that represent the path
    :param hexagon_nodes: all nodes on the hexagon graph. This dataframe is used to determine
    the edge geometries on the hexagon graph.
    """

    edges_list = []
    for source_node, target_node in zip(path_node_indices, path_node_indices[1:]):
        edge_data = graph.get_edge_data(source_node, target_node)

        # As "vanilla" hexagon edges do not have dataclasses as edge attribute, the data must
        # be constructed manually. For all other edges (i.e., piperamming and hexagon connection
        # edges), the data can be converted from the dataclass.
        if isinstance(edge_data, BaseWeightedEdgeInfo):
            edge_meta_data = asdict(edge_data)
        else:
            edge_id = graph.edge_indices_from_endpoints(source_node, target_node)[0]
            edge_linestring = shapely.LineString(
                [
                    get_hexagon_node_geometry(hexagon_nodes, node_id=source_node),
                    get_hexagon_node_geometry(hexagon_nodes, node_id=target_node),
                ]
            )
            edge_meta_data = dict(
                edge_id=edge_id,
                weight=edge_data.weight,
                length=round(edge_linestring.length, 2),
                connects_height_levels=False,
                geometry=edge_linestring,
            )

        edges_list.append(edge_meta_data)

    hexagon_path_geometries = gpd.GeoDataFrame(data=edges_list, crs=Config.CRS)
    hexagon_path_geometries.loc[:, "connects_height_levels"] = hexagon_path_geometries.loc[
        :, "connects_height_levels"
    ].fillna(False)
    return hexagon_path_geometries


def convert_hexagon_edges_to_gdf(graph: rx.PyGraph, nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert all edges in a Hexagon graph to a GeoDataframe.

    Note: when supplying large graphs, this function can take some time to complete due to the geometry creation
    of all edges in the graph.

    :param graph: graph to convert edges for
    :param nodes: all nodes in the graph as a geodataframe containing the source and target geometries
    :return: geodataframe with all edges, edge weights and geometries from the input graph
    """

    edge_weight_map = graph.edge_index_map()
    source_nodes = [source_node for source_node, _, _ in edge_weight_map.values()]
    target_nodes = [target_node for _, target_node, _ in edge_weight_map.values()]

    weights = []
    connects_height_levels = []
    for _, _, edge_data in edge_weight_map.values():
        weights.append(edge_data.weight)
        if isinstance(edge_data, HexagonConnectionEdgeInfo):
            connects_height_levels.append(edge_data.connects_height_levels)
        else:
            connects_height_levels.append(False)

    source_coordinates = nodes.loc[source_nodes].get_coordinates().values
    target_coordinates = nodes.loc[target_nodes].get_coordinates().values
    edge_geometries = shapely.linestrings(np.stack([source_coordinates, target_coordinates], axis=1))

    return gpd.GeoDataFrame(
        {
            "source_node": source_nodes,
            "target_node": target_nodes,
            "weight": weights,
            "connects_height_levels": connects_height_levels,
            "geometry": edge_geometries,
        },
        crs=Config.CRS,
    )
