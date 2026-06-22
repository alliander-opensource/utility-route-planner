# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import rustworkx as rx
import shapely

from utility_route_planner.models.multilayer_network.graph_datastructures import OSMNodeInfo, OSMEdgeInfo


def create_osm_edge_info(osm_id: int, start_node: OSMNodeInfo, end_node: OSMNodeInfo) -> OSMEdgeInfo:
    return OSMEdgeInfo(osm_id=osm_id, geometry=shapely.LineString([start_node.geometry, end_node.geometry]))


def build_osm_test_graph(
    nodes: list[tuple[int, tuple[float, float]]],
    edges: list[tuple[int, int, int]],
) -> rx.PyGraph:
    """Build a simple OSM test graph."""
    graph = rx.PyGraph()
    node_map: dict[int, OSMNodeInfo] = {}

    for osm_id, (x, y) in nodes:
        node = OSMNodeInfo(osm_id=osm_id, geometry=shapely.Point(x, y))
        node.node_id = graph.add_node(node)
        node_map[osm_id] = node

    for from_osm_id, to_osm_id, edge_osm_id in edges:
        src = node_map[from_osm_id]
        dst = node_map[to_osm_id]
        edge_info = create_osm_edge_info(edge_osm_id, src, dst)
        edge_info.edge_id = graph.add_edge(src.node_id, dst.node_id, edge_info)

    return graph
