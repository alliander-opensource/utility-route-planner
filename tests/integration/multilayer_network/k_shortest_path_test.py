# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import networkx as nx
import rustworkx as rx
from shapely import Polygon, Point

from settings import Config
from utility_route_planner.models.multilayer_network.k_shortest_path_route_planner import KShortestPathRoutePlanner
from utility_route_planner.models.multilayer_network.osm_graph_preprocessing import OSMGraphPreprocessor
from utility_route_planner.util.geo_utilities import osm_graph_to_gdfs
from utility_route_planner.util.write import write_results_to_geopackage


class TestKShortestPaths:
    out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT
    debug: bool = True

    @pytest.fixture()
    def project_area(self) -> Polygon:
        return Polygon(
            [
                Point(174730.23, 451038.86),
                Point(175171.37, 451080.10),
                Point(175276.86, 450903.64),
                Point(174674.60, 450792.40),
                Point(174730.23, 451038.86),
            ]
        )

    @pytest.fixture
    def route_planner(self) -> KShortestPathRoutePlanner:
        return KShortestPathRoutePlanner(similarity_threshold=0.5)

    @pytest.fixture()
    def preprocessed_graph(self, load_osm_graph_pickle: nx.MultiDiGraph, project_area: Polygon) -> rx.PyGraph:
        osm_graph_preprocessor = OSMGraphPreprocessor(load_osm_graph_pickle, project_area)
        return osm_graph_preprocessor.preprocess_graph()

    def test_shortest_paths(self, route_planner: KShortestPathRoutePlanner, preprocessed_graph: rx.PyGraph):
        k = 3
        source_node, target_node = 29, 0

        routes = route_planner.find_k_routes(preprocessed_graph, source=source_node, target=target_node, k=k)

        if self.debug:
            nodes, edges = osm_graph_to_gdfs(preprocessed_graph)
            write_results_to_geopackage(self.out, nodes, "pytest_k_shortest_paths_osm_graph_nodes", overwrite=True)
            write_results_to_geopackage(self.out, edges, "pytest_k_shortest_paths_osm_graph_edges", overwrite=True)

            for i, route in enumerate(routes):
                write_results_to_geopackage(
                    self.out, route, f"pytest_k_shortest_paths_osm_graph_route_{i + 1}", overwrite=True
                )
