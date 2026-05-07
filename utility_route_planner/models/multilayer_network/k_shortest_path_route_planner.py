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

    def __init__(self, similarity_threshold: float):
        self.similarity_threshold = similarity_threshold

    def find_k_routes(self, graph: rx.PyGraph, source: int, target: int, k: int):
        self.onepass(graph, source, target, k)

    def onepass(self, graph: rx.PyGraph, source: int, target: int, k: int):
        initial_route = rx.dijkstra_shortest_paths(graph, source, target, weight_fn=lambda x: x.length)[target]
        result_routes = [list(initial_route)]

        # Init candidate queue from starting node
        candidate_route_queue: list = []
        heapq.heappush(candidate_route_queue, (0.0, source, [source]))

        while candidate_route_queue and len(result_routes) < k:
            cost, node_n, candidate_path = heapq.heappop(candidate_route_queue)

            # Target node is reached. As similarity is already checked when adding the path to the queue, this serves
            # as a valid path and can therefore be added to the results
            if node_n == target:
                result_routes.append(candidate_path)

                # Prune queue by removing candidate paths that do not pass the sim check
                pruned_queue = []
                for queue_path_cost, queue_node, queue_path in candidate_route_queue:
                    if not self.path_is_similar(queue_path, result_routes):
                        pruned_queue.append((queue_path_cost, queue_node, queue_path))
                # Update queue with pruned queue
                heapq.heapify(pruned_queue)
                candidate_route_queue = pruned_queue

            else:
                for neighbour_node in graph.neighbors(node_n):
                    # Skip neighbours already in the path as this would lead to the same paths flooding the queue
                    if neighbour_node in candidate_path:
                        continue

                    neighbour_edge_data = graph.get_edge_data(node_n, neighbour_node)
                    potential_candidate_path = candidate_path + [neighbour_node]
                    potential_candidate_path_cost = neighbour_edge_data.geometry.length

                    if not self.path_is_similar(potential_candidate_path, result_routes):
                        heapq.heappush(
                            candidate_route_queue,
                            (potential_candidate_path_cost, neighbour_node, potential_candidate_path),
                        )

    def path_is_similar(self, queue_path: list[int], result_routes: list[list[int]]) -> bool:
        return True

    @staticmethod
    def _convert_osm_route_to_linestring(graph: rx.PyGraph, route: rx.NodeIndices | list[int]) -> LineString:
        linestrings = []
        for u, v in zip(route[1:], route[:-1]):
            linestrings.append(graph.get_edge_data(u, v).geometry)
        route_linestring = linemerge(linestrings)
        return route_linestring
