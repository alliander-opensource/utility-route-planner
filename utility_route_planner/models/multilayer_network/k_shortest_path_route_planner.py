# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import rustworkx as rx
from shapely import LineString
from shapely.ops import linemerge


class KShortestPathRoutePlanner:
    def find_k_routes(self, graph: rx.PyGraph, source: int, target: int, k: int):
        route = rx.dijkstra_shortest_paths(graph, source, target, weight_fn=lambda x: x.length)
        route_linestring = self._convert_osm_route_to_linestring(graph, route[target])
        return route_linestring
        # Expand graph and apply pruning using queuing strategy as proposed
        # in https://link.springer.com/article/10.1007/s00778-020-00604-x

    @staticmethod
    def _convert_osm_route_to_linestring(graph: rx.PyGraph, route: rx.NodeIndices) -> LineString:
        linestrings = []
        for u, v in zip(route[1:], route[:-1]):
            linestrings.append(graph.get_edge_data(u, v).geometry)
        route_linestring = linemerge(linestrings)
        return route_linestring
