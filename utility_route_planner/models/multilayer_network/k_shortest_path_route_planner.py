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

    def find_k_routes(self, graph: rx.PyGraph, source: int, target: int, k: int) -> list[LineString]:
        result_routes = self.onepass(graph, source, target, k)
        result_linestrings = [self._convert_osm_route_to_linestring(graph, route) for route in result_routes]
        return result_linestrings

    def onepass(self, graph: rx.PyGraph, source: int, target: int, k: int):
        initial_route = rx.dijkstra_shortest_paths(graph, source, target, weight_fn=lambda x: x.length)[target]
        result_routes = [list(initial_route)]

        # Init candidate queue from starting node
        candidate_route_queue: list = []
        heapq.heappush(candidate_route_queue, (0.0, source, [source]))

        while candidate_route_queue and len(result_routes) < k:
            candidate_path_cost, node_n, candidate_path = heapq.heappop(candidate_route_queue)

            # Target node is reached. As similarity is already checked when adding the path to the queue, this serves
            # as a valid path and can therefore be added to the results
            if node_n == target:
                result_routes.append(candidate_path)

                # Prune queue by removing queue paths that are similar to
                pruned_queue = []
                for queue_path_cost, queue_node, queue_path in candidate_route_queue:
                    if all(self.path_not_similar(graph, queue_path, result_route) for result_route in result_routes):
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

                    potential_candidate_path_cost = candidate_path_cost + neighbour_edge_data.length

                    # Only add the candidate path to the queue if is dissimilar to the already found routes
                    if all(
                        self.path_not_similar(graph, potential_candidate_path, result_route)
                        for result_route in result_routes
                    ):
                        heapq.heappush(
                            candidate_route_queue,
                            (potential_candidate_path_cost, neighbour_node, potential_candidate_path),
                        )
        return result_routes

    def path_not_similar(self, graph: rx.PyGraph, candidate_route: list[int], result_route: list[int]) -> bool:
        # TODO: move everything to edge ids?
        # TODO: candidate edge do only have to be retrieved once? Result route can also be stored as edge ids already?
        candidate_edge_ids = {
            graph.edge_indices_from_endpoints(u, v)[0] for u, v in zip(candidate_route[1:], candidate_route[:-1])
        }
        result_edge_ids = {
            graph.edge_indices_from_endpoints(u, v)[0] for u, v in zip(result_route[1:], result_route[:-1])
        }

        shared_edges = candidate_edge_ids.intersection(result_edge_ids)

        shared_edge_cost = sum([graph.get_edge_data_by_index(edge_id).length for edge_id in shared_edges])
        result_edge_cost = sum([graph.get_edge_data_by_index(edge_id).length for edge_id in result_edge_ids])

        return shared_edge_cost / result_edge_cost <= self.similarity_threshold

    @staticmethod
    def _convert_osm_route_to_linestring(graph: rx.PyGraph, route: rx.NodeIndices | list[int]) -> LineString:
        linestrings = []
        for u, v in zip(route[1:], route[:-1]):
            linestrings.append(graph.get_edge_data(u, v).geometry)
        route_linestring = linemerge(linestrings)
        return route_linestring
