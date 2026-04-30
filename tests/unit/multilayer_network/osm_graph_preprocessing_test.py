# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import rustworkx as rx

from settings import Config
from networkx import MultiDiGraph, MultiGraph
from pyproj import CRS
import shapely

from utility_route_planner.util.geo_utilities import osm_graph_to_gdfs
from utility_route_planner.models.multilayer_network.osm_graph_preprocessing import (
    OSMGraphPreprocessor,
)
from utility_route_planner.models.multilayer_network.graph_datastructures import OSMEdgeInfo, OSMNodeInfo
from utility_route_planner.util.write import write_results_to_geopackage


class TestOSMGraphPreprocessor:
    @pytest.fixture
    def unprocessed_directed_graph(self):
        graph = MultiDiGraph()
        graph.graph["crs"] = CRS(Config.CRS)

        graph.add_node(0, osmid=0, x=0, y=0)
        graph.add_node(1, osmid=1, x=0, y=1)
        graph.add_node(2, osmid=2, x=0, y=-2)

        # Two-way edges between node 0 and 1
        graph.add_edge(0, 1, 0, osmid=0, geometry=shapely.LineString([(0, 0), (0, 1)]))
        graph.add_edge(1, 0, 0, osmid=1, geometry=shapely.LineString([(0, 1), (0, 0)]))

        # Oneway edge from node 0 to 2
        graph.add_edge(0, 2, 0, osmid=2, geometry=shapely.LineString([(0, 0), (0, -2)]))

        return graph

    def test_graph_is_made_undirected(self, unprocessed_directed_graph: MultiDiGraph):
        graph_preprocessor = OSMGraphPreprocessor(unprocessed_directed_graph)
        preprocessed_graph = graph_preprocessor.preprocess_graph()

        assert isinstance(preprocessed_graph, rx.PyGraph)
        assert preprocessed_graph.num_edges() == 2

        self.check_graph_properties(preprocessed_graph, graph_preprocessor.nx_graph)

    def test_duplicate_edges_are_removed(self, unprocessed_directed_graph: MultiDiGraph):
        # Add duplicate edge
        unprocessed_directed_graph.add_edge(0, 1, 1, osmid=0, geometry=shapely.LineString([(0, 0), (0, 1)]))
        graph_preprocessor = OSMGraphPreprocessor(unprocessed_directed_graph)
        preprocessed_graph = graph_preprocessor.preprocess_graph()

        # For all edges in the preprocessed graph, the key should be 0. Keys > 0 imply duplicate edges
        assert preprocessed_graph.num_edges() == 2

        self.check_graph_properties(preprocessed_graph, graph_preprocessor.nx_graph)

    def test_osm_pickle(self, load_osm_graph_pickle: MultiGraph):
        graph_preprocessor = OSMGraphPreprocessor(load_osm_graph_pickle)
        preprocessed_graph = graph_preprocessor.preprocess_graph()

        self.check_graph_properties(preprocessed_graph, graph_preprocessor.nx_graph)

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            OSMGraphPreprocessor(None).preprocess_graph()
        graph = MultiDiGraph()
        with pytest.raises(ValueError):
            OSMGraphPreprocessor(graph).preprocess_graph()
        graph.add_node(0, osmid=0, geometry=shapely.LineString([(0, 0), (0, 1)]))
        with pytest.raises(ValueError):
            OSMGraphPreprocessor(graph).preprocess_graph()

    def test_osm_graph_to_gdfs_conversion(self, load_osm_graph_pickle: MultiGraph):
        graph_preprocessor = OSMGraphPreprocessor(load_osm_graph_pickle)
        preprocessed_graph = graph_preprocessor.preprocess_graph()

        self.check_gdf_properties(preprocessed_graph)

    def test_osm_graph_to_gdfs_conversion_with_changes(self, load_osm_graph_pickle: MultiGraph):
        graph_preprocessor = OSMGraphPreprocessor(load_osm_graph_pickle)
        preprocessed_graph = graph_preprocessor.preprocess_graph()

        preprocessed_graph.remove_node(2)

        new_node_1 = OSMNodeInfo(osm_id=123, geometry=shapely.Point(1, 1))
        idx_1 = preprocessed_graph.add_node(new_node_1)
        assert idx_1 == 2  # must be equal to the removed node id
        preprocessed_graph[idx_1].node_id = idx_1

        new_node_2 = OSMNodeInfo(osm_id=124, geometry=shapely.Point(1, 2))
        idx_2 = preprocessed_graph.add_node(new_node_2)
        preprocessed_graph[idx_2].node_id = idx_2
        new_node_3 = OSMNodeInfo(osm_id=125, geometry=shapely.Point(1, 3))
        idx_3 = preprocessed_graph.add_node(new_node_3)
        preprocessed_graph[idx_3].node_id = idx_3

        preprocessed_graph.remove_edge(*preprocessed_graph.get_edge_endpoints_by_index(40))
        preprocessed_graph.remove_edge(*preprocessed_graph.get_edge_endpoints_by_index(41))

        idx_5 = preprocessed_graph.add_edge(
            0,
            1,
            OSMEdgeInfo(
                osm_id=126,
                geometry=shapely.LineString([preprocessed_graph[0].geometry, preprocessed_graph[1].geometry]),
            ),
        )
        preprocessed_graph.get_edge_data(0, 1).edge_id = idx_5
        idx_6 = preprocessed_graph.add_edge(
            idx_2,
            idx_3,
            OSMEdgeInfo(
                osm_id=127,
                geometry=shapely.LineString([preprocessed_graph[idx_2].geometry, preprocessed_graph[idx_3].geometry]),
            ),
        )
        preprocessed_graph.get_edge_data(idx_2, idx_3).edge_id = idx_6

        self.check_gdf_properties(preprocessed_graph)

    def test_osm_graph_to_gdfs_empty_return(self, load_osm_graph_pickle: MultiGraph):
        gdf_nodes, gdf_edges = osm_graph_to_gdfs(rx.PyGraph())
        assert len(gdf_nodes) == 0
        assert len(gdf_edges) == 0

    @staticmethod
    def check_gdf_properties(rx_graph: rx.PyGraph):
        gdf_nodes, gdf_edges = osm_graph_to_gdfs(rx_graph)
        assert len(gdf_nodes) == rx_graph.num_nodes()
        assert len(gdf_edges) == rx_graph.num_edges()

        for node in rx_graph.nodes():
            assert gdf_nodes.loc[node.node_id].osm_id == node.osm_id
            assert gdf_nodes.loc[node.node_id].geometry == node.geometry

        for edge in rx_graph.edges():
            assert gdf_edges.loc[edge.edge_id].osm_id == edge.osm_id
            assert gdf_edges.loc[edge.edge_id].geometry == edge.geometry
            assert gdf_edges.loc[edge.edge_id].length == edge.length

    @staticmethod
    def check_graph_properties(rx_graph: rx.PyGraph, nx_graph: MultiGraph):
        assert nx_graph.number_of_nodes() == rx_graph.num_nodes()
        assert nx_graph.number_of_edges() == rx_graph.num_edges()

        nx_nodes = nx_graph.nodes(data=True)
        for node_id in rx_graph.node_indices():
            node = rx_graph[node_id]
            # Check if the properties of the node are the same as the nx_graph
            assert isinstance(node, OSMNodeInfo)
            assert nx_nodes[node.osm_id].get("x") == node.geometry.x
            assert nx_nodes[node.osm_id].get("y") == node.geometry.y

            # Check if we have the same edges as neighbours in both graphs
            rx_adjacent_edges = rx_graph.adj(node_id)
            nx_adjacent_edges = nx_graph.edges(node.osm_id, data=True)
            assert len(rx_adjacent_edges) == len(nx_adjacent_edges)

            # Check if the edge properties are the same
            nx_edge_props = set((i[2]["geometry"], round(i[2]["length"], 2), i[2]["osmid"]) for i in nx_adjacent_edges)
            rx_edge_props = set(
                (rx_edge.geometry, rx_edge.length, rx_edge.osm_id) for rx_edge in rx_adjacent_edges.values()
            )
            assert rx_edge_props == nx_edge_props

    def test_clip_graph_to_polygon(self, load_osm_graph_pickle: MultiDiGraph, debug: bool = False):
        polygon = shapely.Point(174968.78, 451010.45).buffer(
            100
        )  # Note, this needs to be within the pickled OSM graph.

        graph_preprocessor = OSMGraphPreprocessor(load_osm_graph_pickle, polygon, debug)
        n_nodes_old, n_edges_old = (
            graph_preprocessor.nx_graph.number_of_nodes(),
            graph_preprocessor.nx_graph.number_of_edges(),
        )
        rx_graph = graph_preprocessor.preprocess_graph()

        assert rx_graph.num_nodes() == 11 < n_nodes_old
        assert rx_graph.num_edges() == 10 < n_edges_old

        nodes, edges = osm_graph_to_gdfs(rx_graph)

        assert all(nodes.within(polygon))
        assert all(edges.within(polygon))
        if debug:
            write_results_to_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, nodes, "pytest_clip_nodes")
            write_results_to_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, edges, "pytest_clip_edges")
