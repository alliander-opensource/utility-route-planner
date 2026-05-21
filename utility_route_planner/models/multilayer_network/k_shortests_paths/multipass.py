# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import heapq

import rustworkx as rx

from utility_route_planner.models.multilayer_network.k_shortests_paths.k_shortest_path_route_planner import (
    KShortestPathAlgorithm,
)


class MultiPassPlanner(KShortestPathAlgorithm):
    def __init__(self, similarity_threshold: float):
        super().__init__(similarity_threshold)

    def find_k_routes(self, graph: rx.PyGraph, source: int, target: int, k: int) -> list[list[int]]:
        initial_route = rx.dijkstra_shortest_paths(graph, source, target, weight_fn=lambda x: x.length)[target]
        result_routes = [list(initial_route)]

        previous_result_size = 0
        # Continue while the number of result routes is < k and new routes are being found in each round
        while k > len(result_routes) > previous_result_size:
            previous_result_size = len(result_routes)
            # Init candidate queue from starting node at each round
            candidate_route_queue: list = []
            heapq.heappush(candidate_route_queue, (0.0, source, [source]))

            # Associate each node with empty set of labels
            node_labels: dict[int, list[tuple[float, list[int]]]] = {}

            while candidate_route_queue:
                candidate_path_cost, node_n, candidate_path = heapq.heappop(candidate_route_queue)

                if node_n == target:
                    result_routes.append(candidate_path)
                    break
                else:
                    for neighbour_node in graph.neighbors(node_n):
                        neighbour_edge_data = graph.get_edge_data(node_n, neighbour_node)
                        potential_candidate_path = candidate_path + [neighbour_node]

                        potential_candidate_path_cost = candidate_path_cost + neighbour_edge_data.length

                        if any(
                            self.lemma_2_similarity(graph, potential_candidate_path, result_route)
                            >= self.similarity_threshold
                            for result_route in result_routes
                        ):
                            continue
                        elif self.lemma_3_similarity(
                            neighbour_node,
                            potential_candidate_path,
                            potential_candidate_path_cost,
                            node_labels,
                            result_routes,
                            graph,
                        ):
                            continue
                        else:
                            heapq.heappush(
                                candidate_route_queue,
                                (potential_candidate_path_cost, neighbour_node, potential_candidate_path),
                            )

                            if neighbour_node not in node_labels:
                                node_labels[neighbour_node] = []
                            node_labels[neighbour_node].append(
                                (potential_candidate_path_cost, potential_candidate_path)
                            )
        return result_routes

    def lemma_2_similarity(self, graph: rx.PyGraph, candidate_route: list[int], result_route: list[int]) -> float:
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

        return shared_edge_cost / result_edge_cost

    def lemma_3_similarity(
        self,
        neighbour_node: int,
        candidate_route: list[int],
        candidate_route_cost: float,
        node_labels: dict[int, list[tuple[float, list[int]]]],
        result_routes: list,
        graph: rx.PyGraph,
    ) -> bool:
        if neighbour_node not in node_labels:
            return False
        for label_cost, label_path in node_labels[neighbour_node]:
            if self.dominates(label_path, label_cost, candidate_route, candidate_route_cost, result_routes, graph):
                return True
        return False

    def dominates(
        self,
        previous_route: list[int],
        previous_route_cost: float,
        candidate_route: list[int],
        candidate_route_cost: float,
        result_routes: list,
        graph: rx.PyGraph,
    ):
        # Label route has a higher cost than the new candidate route, this means that the candidate route is not
        # dominated by the label route.
        if previous_route_cost <= candidate_route_cost:
            return True

        # If the previous route has less similarity with any of the result routes compared to the candidate route, this
        # means that the candidate route is dominated by the previous route
        for result_route in result_routes:
            if self.lemma_2_similarity(graph, previous_route, result_route) <= self.lemma_2_similarity(
                graph, candidate_route, result_route
            ):
                return True
        return False

    def remove_dominated_routes(
        self,
        neighbour_node: int,
        new_route: list[int],
        new_route_cost: float,
        node_labels: dict[int, list[tuple[float, list[int]]]],
        queue: list,
        result_routes: list,
        graph: rx.PyGraph,
    ):
        # Remove all paths which are dominated by the new result route for this neighbour node
        if neighbour_node in node_labels:
            node_labels[neighbour_node] = [
                (cost, path)
                for cost, path in node_labels[neighbour_node]
                if not self.dominates(new_route, new_route_cost, path, cost, result_routes, graph)
            ]

        # Remove dominated routes from the queue as well
        queue[:] = [
            entry
            for entry in queue
            if not (
                entry[1] == neighbour_node
                and self.dominates(new_route, new_route_cost, entry[2], entry[0], result_routes, graph)
            )
        ]
        heapq.heapify(queue)
