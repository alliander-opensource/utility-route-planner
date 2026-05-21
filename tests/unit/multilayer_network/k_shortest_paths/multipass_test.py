# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import rustworkx as rx
from shapely import LineString, Point

from settings import Config
from utility_route_planner.models.multilayer_network.graph_datastructures import OSMEdgeInfo
from utility_route_planner.models.multilayer_network.k_shortests_paths.multipass import MultiPassPlanner


@pytest.fixture
def multipass_planner() -> MultiPassPlanner:
    return MultiPassPlanner(similarity_threshold=Config.K_SHORTEST_PATH_SIMILARITY_THRESHOLD)


class TestOneWaySimilarity:
    @pytest.fixture()
    def graph(self) -> rx.PyGraph:
        graph = rx.PyGraph()
        graph.add_nodes_from(list(range(6)))
        edge_ids = graph.add_edges_from(
            [
                (0, 1, OSMEdgeInfo(osm_id=0, geometry=LineString([Point(0, 0), Point(1, -0.5)]))),
                (0, 2, OSMEdgeInfo(osm_id=1, geometry=LineString([Point(0, 0), Point(1, 0.5)]))),
                (2, 1, OSMEdgeInfo(osm_id=2, geometry=LineString([Point(1, 0.5), Point(1, -0.5)]))),
                (1, 3, OSMEdgeInfo(osm_id=3, geometry=LineString([Point(1, -0.5), Point(2, 0.5)]))),
                (3, 5, OSMEdgeInfo(osm_id=4, geometry=LineString([Point(2, 0.5), Point(3, 0)]))),
                (2, 4, OSMEdgeInfo(osm_id=5, geometry=LineString([Point(1, 0.5), Point(2, -0.5)]))),
                (4, 5, OSMEdgeInfo(osm_id=6, geometry=LineString([Point(2, -0.5), Point(3, 0)]))),
            ]
        )
        [edge.set_edge_id(edge_id) for edge, edge_id in zip(graph.edges(), edge_ids)]

        return graph

    @pytest.fixture()
    def accepted_route(self) -> list[int]:
        return [0, 2, 4, 5]

    def test_partial_route_with_overlap_exceeds_threshold(
        self, multipass_planner: MultiPassPlanner, graph: rx.PyGraph, accepted_route: list[int]
    ):
        """
        Validates that a partial candidate route with exact overlap (node 0, 2 and 4 are in the accepted route  as well)
        results in route similarity which exceeds the threshold.
        """
        candidate_partial_route = [0, 2, 4]
        result_similarity = multipass_planner.one_way_path_similarity(graph, candidate_partial_route, accepted_route)
        assert result_similarity > Config.K_SHORTEST_PATH_SIMILARITY_THRESHOLD

    def test_partial_route_with_overlap_does_not_exceed_threshold(
        self, multipass_planner: MultiPassPlanner, graph: rx.PyGraph, accepted_route: list[int]
    ):
        """
        Validates that a starting candidate partial route does not exceed the threshold yet, even though all nodes in
        the candidate route are part of the accepted route as well. As it is a starting route with only a single shared
        edge, the threshold must not be exceeded yet to allow this candidate route to differ from the accepted route(s)
        when expanding further.
        """

        candidate_partial_route = [0, 2]
        result_similarity = multipass_planner.one_way_path_similarity(graph, candidate_partial_route, accepted_route)
        assert result_similarity < Config.K_SHORTEST_PATH_SIMILARITY_THRESHOLD

    def test_similar_length_different_edges_does_not_exceed_threshold(
        self, multipass_planner: MultiPassPlanner, graph: rx.PyGraph, accepted_route: list[int]
    ):
        """
        Validates that a candidate route with a similar length as the accepted route must not exceed the similarity
        threshold as it uses different edges.
        """
        candidate_route = [0, 1, 3, 5]
        result_similarity = multipass_planner.one_way_path_similarity(graph, candidate_route, accepted_route)
        assert result_similarity < Config.K_SHORTEST_PATH_SIMILARITY_THRESHOLD

    def test_longer_length_but_dissimilar_edges_does_not_exceed_threshold(
        self, multipass_planner: MultiPassPlanner, graph: rx.PyGraph, accepted_route: list[int]
    ):
        """
        Validates that a candidate route which is slightly longer than the accepted route must not exceed the similarity
        threshold as it uses different edges.
        """
        candidate_route = [0, 2, 1, 3, 5]
        result_similarity = multipass_planner.one_way_path_similarity(graph, candidate_route, accepted_route)
        assert result_similarity < Config.K_SHORTEST_PATH_SIMILARITY_THRESHOLD
