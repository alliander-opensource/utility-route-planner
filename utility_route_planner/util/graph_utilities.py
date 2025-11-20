# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import shapely

from utility_route_planner.models.multilayer_network.graph_datastructures import OSMNodeInfo, OSMEdgeInfo


def create_osm_edge_info(osm_id: int, start_node: OSMNodeInfo, end_node: OSMNodeInfo) -> OSMEdgeInfo:
    return OSMEdgeInfo(osm_id=osm_id, geometry=shapely.LineString([start_node.geometry, end_node.geometry]))
