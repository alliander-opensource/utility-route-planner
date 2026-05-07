# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import heapq

import rustworkx as rx
from shapely import LineString
from shapely.ops import linemerge


class KShortestPathRoutePlanner:
    """
    Compute k-shortest paths on an OSM spatial graph using methods as proposed in:
    Chondrogiannis, T., Bouros, P., Gamper, J. et al. Finding k-shortest paths with limited overlap.
    The VLDB Journal 29, 1023–1047 (2020). https://doi.org/10.1007/s00778-020-00604-x
    """

    def find_k_routes(self, graph: rx.PyGraph, source: int, target: int, k: int):
        self.onepass(graph, source, target, k)

    def onepass(self, graph: rx.PyGraph, source: int, target: int, k: int):
        start_route = rx.dijkstra_shortest_paths(graph, source, target, weight_fn=lambda x: x.length)
        start_route_nodes = start_route[target]

        self._compute_candidate_routes(graph, start_route_nodes, target)

    def _compute_candidate_routes(self, graph: rx.PyGraph, start_route_nodes: rx.NodeIndices, target: int) -> list:
        """
        Compute candidate routes given the initial shortest path from source to target. Each candidate route is computed
        by iterating over all edges u-v in the shortest path and:
        1: Remove the edge temporarily from the graph.
        2. Starting from u, compute the shortest branch path from u to the target node. As u-v is temporarily removed
        from the graph, this forces the candidate_route to be different from the initial shortest path.
        3. The candidate route is constructed by concatenating shortest_path_start_node-u to u-target_node.
        4. The candidate route and length are stored in the min_priority queue which is sorted on route length. This
           guarantees the evaluation of shortest paths first when computing route similarity.
        5. The removed edge is readded to the graph.

        This process is repeated for all nodes on the initial shortest path, except for the target node (node_-1)
        """

        min_length_queue: list = []
        for i, u in enumerate(start_route_nodes[:-1]):
            v = start_route_nodes[i + 1]
            removed_edge_data = graph.get_edge_data(u, v)
            graph.remove_edge(u, v)

            branch_path_nodes = rx.dijkstra_shortest_paths(graph, u, target, weight_fn=lambda x: x.length)[target]
            candidate_path_nodes = list(start_route_nodes[:i]) + list(branch_path_nodes)
            candidate_linestring = self._convert_osm_route_to_linestring(graph, candidate_path_nodes)
            heapq.heappush(min_length_queue, (candidate_linestring.length, candidate_path_nodes))

            graph.add_edge(u, v, removed_edge_data)

        return min_length_queue

    @staticmethod
    def _convert_osm_route_to_linestring(graph: rx.PyGraph, route: rx.NodeIndices | list[int]) -> LineString:
        linestrings = []
        for u, v in zip(route[1:], route[:-1]):
            linestrings.append(graph.get_edge_data(u, v).geometry)
        route_linestring = linemerge(linestrings)
        return route_linestring
