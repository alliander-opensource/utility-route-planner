# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

import rustworkx as rx
from shapely import LineString
from shapely.ops import linemerge


class KShortestPathAlgorithm(ABC):
    def __init__(self, similarity_threshold: float):
        self.similarity_threshold = similarity_threshold

    @abstractmethod
    def find_k_routes(self, graph: rx.PyGraph, source: int, target: int, k: int) -> list[list[int]]:
        pass


class KShortestPathRoutePlanner:
    """
    Compute k-shortest paths on an OSM spatial graph using methods as proposed in:
    Chondrogiannis, T., Bouros, P., Gamper, J. et al. Finding k-shortest paths with limited overlap.
    The VLDB Journal 29, 1023–1047 (2020). https://doi.org/10.1007/s00778-020-00604-x
    """

    def __init__(self, shortest_path_algorithm: KShortestPathAlgorithm):
        self.shortest_path_algorithm = shortest_path_algorithm

    def find_k_routes(self, graph: rx.PyGraph, source: int, target: int, k: int) -> list[LineString]:
        result_routes = self.shortest_path_algorithm.find_k_routes(graph, source, target, k)
        result_linestrings = [self._convert_osm_route_to_linestring(graph, route) for route in result_routes]
        return result_linestrings

    @staticmethod
    def _convert_osm_route_to_linestring(graph: rx.PyGraph, route: rx.NodeIndices | list[int]) -> LineString:
        linestrings = []
        for u, v in zip(route[1:], route[:-1]):
            linestrings.append(graph.get_edge_data(u, v).geometry)
        route_linestring = linemerge(linestrings)
        return route_linestring
