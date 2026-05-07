# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import geopandas as gpd
import pytest
import networkx as nx
import rustworkx as rx

from settings import Config
from utility_route_planner.models.multilayer_network.k_shortest_path_route_planner import KShortestPathRoutePlanner
from utility_route_planner.models.multilayer_network.osm_graph_preprocessing import OSMGraphPreprocessor
from utility_route_planner.util.geo_utilities import osm_graph_to_gdfs
from utility_route_planner.util.write import write_results_to_geopackage


class TestKShortestPaths:
    out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT
    debug: bool = True

    @pytest.fixture
    def route_planner(self) -> KShortestPathRoutePlanner:
        return KShortestPathRoutePlanner()

    @pytest.fixture()
    def preprocessed_graph(self, load_osm_graph_pickle: nx.MultiDiGraph) -> rx.PyGraph:
        project_area = gpd.read_file(
            Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA
        ).geometry.iloc[0]

        osm_graph_preprocessor = OSMGraphPreprocessor(load_osm_graph_pickle, project_area)
        return osm_graph_preprocessor.preprocess_graph()

    def test_shortest_paths(self, route_planner: KShortestPathRoutePlanner, preprocessed_graph: rx.PyGraph):
        k = 3
        source_node, target_node = 328, 35

        route = route_planner.find_k_routes(preprocessed_graph, source=source_node, target=target_node, k=k)

        if self.debug:
            nodes, edges = osm_graph_to_gdfs(preprocessed_graph)
            write_results_to_geopackage(self.out, nodes, "pytest_k_shortest_paths_osm_graph_nodes", overwrite=True)
            write_results_to_geopackage(self.out, edges, "pytest_k_shortest_paths_osm_graph_edges", overwrite=True)
            write_results_to_geopackage(self.out, route, "pytest_k_shortest_paths_osm_graph_route", overwrite=True)
