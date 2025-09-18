# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import geopandas as gpd
import rustworkx as rx
import shapely

from settings import Config
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import convert_hexagon_graph_to_gdfs
from utility_route_planner.util.graph_utilities import create_edge_info
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.multilayer_network.pipe_ramming import GetPotentialPipeRammingCrossings
from utility_route_planner.util.geo_utilities import get_empty_geodataframe, osm_graph_to_gdfs
from utility_route_planner.models.multilayer_network.osm_graph_preprocessing import OSMGraphPreprocessor
from utility_route_planner.models.multilayer_network.graph_datastructures import OSMNodeInfo
from utility_route_planner.util.write import reset_geopackage, write_results_to_geopackage


class TestPipeRamming:
    @pytest.fixture
    def setup_pipe_ramming_example_polygon(self, load_osm_graph_pickle):
        def _setup(project_area=None, debug=False):
            if project_area is None:
                project_area = (
                    gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
                    .iloc[0]
                    .geometry
                )

            osm_graph_preprocessor = OSMGraphPreprocessor(load_osm_graph_pickle)
            osm_graph_preprocessed = osm_graph_preprocessor.preprocess_graph()

            mcda_engine = McdaCostSurfaceEngine(
                Config.RASTER_PRESET_NAME_BENCHMARK, Config.PYTEST_PATH_GEOPACKAGE_MCDA, project_area
            )
            mcda_engine.preprocess_vectors()

            raster_groups = {
                criteria_key: criteria.group for criteria_key, criteria in mcda_engine.raster_preset.criteria.items()
            }
            hexagon_graph_builder = HexagonGraphBuilder(
                mcda_engine.project_area_geometry,
                raster_groups,
                mcda_engine.processed_vectors,
                hexagon_size=0.5,
            )
            cost_surface_graph = hexagon_graph_builder.build_graph()

            if debug:
                osm_nodes, osm_edges = osm_graph_to_gdfs(osm_graph_preprocessed)
                cost_surface_nodes = convert_hexagon_graph_to_gdfs(cost_surface_graph, edges=False)
                out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT
                reset_geopackage(out, truncate=False)
                write_results_to_geopackage(out, osm_nodes, "osm_nodes")
                write_results_to_geopackage(out, osm_edges, "osm_edges")
                write_results_to_geopackage(out, cost_surface_nodes, "cost_surface_nodes")

            return osm_graph_preprocessed, mcda_engine, cost_surface_graph

        return _setup

    def test_create_street_segment_groups(self, debug=False):
        if debug:
            reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)

        osm_graph = rx.PyGraph()

        node1 = OSMNodeInfo(osm_id=1, geometry=shapely.Point(0, 0))
        node2 = OSMNodeInfo(osm_id=2, geometry=shapely.Point(1, 0))
        node3 = OSMNodeInfo(osm_id=3, geometry=shapely.Point(1, -1))
        node4 = OSMNodeInfo(osm_id=4, geometry=shapely.Point(1, -2))
        node5 = OSMNodeInfo(osm_id=5, geometry=shapely.Point(2, 0))
        node6 = OSMNodeInfo(osm_id=6, geometry=shapely.Point(3, 0))
        node7 = OSMNodeInfo(osm_id=7, geometry=shapely.Point(3, 1))
        node8 = OSMNodeInfo(osm_id=8, geometry=shapely.Point(4, 1))
        node9 = OSMNodeInfo(osm_id=9, geometry=shapely.Point(4, 0))
        node10 = OSMNodeInfo(osm_id=10, geometry=shapely.Point(5, 0))
        node11 = OSMNodeInfo(osm_id=11, geometry=shapely.Point(6, 1))
        node12 = OSMNodeInfo(osm_id=12, geometry=shapely.Point(6, -1))

        node_ids = osm_graph.add_nodes_from(
            [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12]
        )
        (
            node1.node_id,
            node2.node_id,
            node3.node_id,
            node4.node_id,
            node5.node_id,
            node6.node_id,
            node7.node_id,
            node8.node_id,
            node9.node_id,
            node10.node_id,
            node11.node_id,
            node12.node_id,
        ) = node_ids

        edges_to_add = [
            (node1.node_id, node2.node_id, create_edge_info(100, node1, node2)),
            (node2.node_id, node3.node_id, create_edge_info(101, node2, node3)),
            (node3.node_id, node4.node_id, create_edge_info(102, node3, node4)),
            (node2.node_id, node5.node_id, create_edge_info(103, node2, node5)),
            (node5.node_id, node6.node_id, create_edge_info(104, node5, node6)),
            (node6.node_id, node7.node_id, create_edge_info(105, node6, node7)),
            (node7.node_id, node8.node_id, create_edge_info(106, node7, node8)),
            (node8.node_id, node9.node_id, create_edge_info(107, node8, node9)),
            (node6.node_id, node9.node_id, create_edge_info(108, node6, node9)),
            (node9.node_id, node10.node_id, create_edge_info(109, node9, node10)),
            (node10.node_id, node11.node_id, create_edge_info(110, node10, node11)),
            (node10.node_id, node12.node_id, create_edge_info(111, node10, node12)),
            (node11.node_id, node12.node_id, create_edge_info(112, node11, node2)),
        ]

        edge_ids = osm_graph.add_edges_from(edges_to_add)
        for edge, edge_id in zip(edges_to_add, edge_ids):
            edge[2].edge_id = edge_id

        # Enable debug for visual debugging in QGIS.
        crossings = GetPotentialPipeRammingCrossings(osm_graph, rx.PyGraph(), get_empty_geodataframe(), debug=debug)
        crossings.create_street_segment_groups()

        edges, nodes = crossings.osm_edges, crossings.osm_nodes

        # Do a sanity check on the grouped edges and nodes.
        assert len(edges) == osm_graph.num_edges()
        assert len(nodes) == osm_graph.num_nodes()

        # Check that the edges are grouped correctly.
        assert edges["group"].nunique() == 7

        group_100 = edges.loc[edges["osm_id"] == 100, "group"].iloc[0]
        assert (edges["group"] == group_100).sum() == 1

        group_101 = edges.loc[edges["osm_id"] == 101, "group"].iloc[0]
        group_102 = edges.loc[edges["osm_id"] == 102, "group"].iloc[0]
        assert group_101 == group_102
        assert (edges["group"] == group_101).sum() == 2

        group_103 = edges.loc[edges["osm_id"] == 103, "group"].iloc[0]
        group_104 = edges.loc[edges["osm_id"] == 104, "group"].iloc[0]
        assert group_103 == group_104
        assert (edges["group"] == group_103).sum() == 2

        group_105 = edges.loc[edges["osm_id"] == 105, "group"].iloc[0]
        group_106 = edges.loc[edges["osm_id"] == 106, "group"].iloc[0]
        group_107 = edges.loc[edges["osm_id"] == 107, "group"].iloc[0]
        assert group_105 == group_106 == group_107
        assert (edges["group"] == group_105).sum() == 3

        group_108 = edges.loc[edges["osm_id"] == 108, "group"].iloc[0]
        assert (edges["group"] == group_108).sum() == 1

        group_109 = edges.loc[edges["osm_id"] == 109, "group"].iloc[0]
        assert (edges["group"] == group_109).sum() == 1

        group_110 = edges.loc[edges["osm_id"] == 110, "group"].iloc[0]
        group_111 = edges.loc[edges["osm_id"] == 111, "group"].iloc[0]
        group_112 = edges.loc[edges["osm_id"] == 112, "group"].iloc[0]
        assert group_110 == group_111 == group_112
        assert (edges["group"] == group_110).sum() == 3

    def test_junction_find_crossings_single_degree_4(self, setup_pipe_ramming_example_polygon, debug=True):
        if debug:
            reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)

        node_id_to_test = 499  # 386
        project_area = shapely.Point(174967.12, 450898.60).buffer(200)

        osm_graph, mcda_engine, cost_surface_graph = setup_pipe_ramming_example_polygon(project_area)

        pipe_ramming = GetPotentialPipeRammingCrossings(osm_graph, cost_surface_graph, debug=debug)
        pipe_ramming.create_street_segment_groups()
        junctions = pipe_ramming.prepare_junction_crossings()
        crossing = pipe_ramming.get_crossing_for_junction(node_id_to_test, junctions.loc[node_id_to_test])
        assert len(crossing) == 3

        # Test our newly found crossing in a shortest path.
        pipe_ramming.add_crossings_to_graph(crossing)
        path = rx.dijkstra_shortest_paths(pipe_ramming.cost_surface_graph, 165602, 139510, lambda x: x.weight)
        path = path[139510]
        path_points = shapely.MultiPoint([pipe_ramming.cost_surface_graph.get_node_data(i).geometry for i in path])
        edges = []
        for current, next_ in zip(path, path[1:]):
            edges.append(pipe_ramming.cost_surface_graph.get_edge_data(current, next_).geometry)
        path_linestring = shapely.MultiLineString(edges)
        assert path_linestring.length == pytest.approx(53, abs=1)
        assert len(path) == 51

        if debug:
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, path_linestring, "pytest_path_result"
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, path_points, "pytest_nodes_result"
            )

    def test_junction_find_crossings_with_custom_obstacles(self, setup_pipe_ramming_example_polygon, debug=False):
        # Obstacle can be a random polygon, check that it is respected.
        pass

    def test_junction_find_crossings_nothing_found(self, setup_pipe_ramming_example_polygon, debug=False):
        # Check that it works when there is no crossing possible.
        pass

    def test_street_segment_group_find_crossings_long(self, setup_pipe_ramming_example_polygon, debug=True):
        if debug:
            reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)
        segment_group_to_cross = 3760

        project_area = shapely.Point(174974, 451093).buffer(200)
        osm_graph, mcda_engine, cost_surface_graph = setup_pipe_ramming_example_polygon(project_area)

        pipe_ramming = GetPotentialPipeRammingCrossings(osm_graph, cost_surface_graph, debug=debug)
        pipe_ramming.create_street_segment_groups()
        pipe_ramming.prepare_junction_crossings()
        segments_of_interest = pipe_ramming.prepare_segment_crossings()
        _ = pipe_ramming.get_crossings_per_segment(
            segment_group_to_cross, segments_of_interest.loc[segment_group_to_cross].geometry
        )

    def test_street_segment_group_find_crossings_short(self):
        pass

    def test_find_all_crossings(self, setup_pipe_ramming_example_polygon, debug=True):
        if debug:
            reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)

        osm_graph, mcda_engine, cost_surface_graph = setup_pipe_ramming_example_polygon()
        pipe_ramming = GetPotentialPipeRammingCrossings(osm_graph, cost_surface_graph, debug=debug)
        _ = pipe_ramming.get_crossings()

    @pytest.mark.skip(reason="First fix the junctions.")
    def test_find_road_crossings(self, setup_pipe_ramming_example_polygon, debug=False):
        if debug:
            reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)

        osm_graph, mcda_engine = setup_pipe_ramming_example_polygon

        obstacles = mcda_engine.processed_vectors["pand"]  # can be expanded with water, trees.
        roads = mcda_engine.processed_vectors["wegdeel"]

        crossings = GetPotentialPipeRammingCrossings(
            osm_graph, Config.PATH_EXAMPLE_RASTER, roads, obstacles, debug=debug
        )
        crossings.get_crossings()
