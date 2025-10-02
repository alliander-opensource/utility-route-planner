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
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine
from utility_route_planner.util.graph_utilities import create_edge_info
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.multilayer_network.pipe_ramming import GetPotentialPipeRammingCrossings
from utility_route_planner.util.geo_utilities import osm_graph_to_gdfs
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

            osm_graph_preprocessor = OSMGraphPreprocessor(load_osm_graph_pickle, project_area)
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
        crossings = GetPotentialPipeRammingCrossings(osm_graph, rx.PyGraph(), debug=debug)
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

    def test_single_junction(self, setup_pipe_ramming_example_polygon, debug=False):
        """For debugging specific junction."""
        if debug:
            reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)

        node_id_to_test = 3
        project_area = shapely.Point(174967.12, 450898.60).buffer(150)

        osm_graph, mcda_engine, cost_surface_graph = setup_pipe_ramming_example_polygon(project_area)

        pipe_ramming = GetPotentialPipeRammingCrossings(osm_graph, cost_surface_graph, debug=debug)
        pipe_ramming.create_street_segment_groups()
        pipe_ramming.prepare_junction_crossings()
        crossings = pipe_ramming.get_crossing_for_junction(
            node_id_to_test,
            pipe_ramming.junctions_of_interests.loc[node_id_to_test].osm_id,
            pipe_ramming.junctions_of_interests.loc[node_id_to_test].geometry,
            pipe_ramming.junctions_of_interests.loc[node_id_to_test].degree,
        )
        assert len(crossings) == 5

        # Test our newly found crossing in a shortest path.
        pipe_ramming.add_crossings_to_graph(crossings)
        start_end = shapely.LineString([(174971.62, 450911.846), (174971.62, 450888.463)])
        multilayer_route_engine = MultilayerRouteEngine(
            pipe_ramming.cost_surface_graph,
            pipe_ramming.osm_graph,
            pipe_ramming.cost_surface_nodes,
            prefix="pytest_junction_",
        )
        multilayer_route_engine.find_route(start_end)

        assert multilayer_route_engine.result_route.length == pytest.approx(24, abs=1)
        assert len([i for i in crossings if i[0] and i[1] in multilayer_route_engine.result_route_node_indices]) == 1, (
            "One of the new edges should be in the path."
        )

    def test_single_street_segment_group(self, setup_pipe_ramming_example_polygon, debug=False):
        """For debugging specific street-segment group."""
        if debug:
            reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)

        segment_group_to_cross = 48

        project_area = shapely.Point(174974, 451093).buffer(150)
        osm_graph, mcda_engine, cost_surface_graph = setup_pipe_ramming_example_polygon(project_area)

        pipe_ramming = GetPotentialPipeRammingCrossings(osm_graph, cost_surface_graph, debug=debug)
        pipe_ramming.suitability_value_obstacles_threshold = 77
        pipe_ramming.create_street_segment_groups()
        pipe_ramming.prepare_junction_crossings()
        segments_of_interest = pipe_ramming.prepare_segment_crossings()
        crossings = pipe_ramming.get_crossings_per_segment(
            segment_group_to_cross, segments_of_interest.loc[segment_group_to_cross].geometry
        )
        pipe_ramming.add_crossings_to_graph(crossings)
        assert len(crossings) == 3

        start_end = shapely.LineString([(174927.5, 451098.452), (174932, 451089.791)])
        multilayer_route_engine = MultilayerRouteEngine(
            pipe_ramming.cost_surface_graph,
            pipe_ramming.osm_graph,
            pipe_ramming.cost_surface_nodes,
            prefix="pytest_junction_",
        )
        multilayer_route_engine.find_route(start_end)

        assert multilayer_route_engine.result_route.length == pytest.approx(12, abs=1)
        assert len([i for i in crossings if i[0] and i[1] in multilayer_route_engine.result_route_node_indices]) == 1, (
            "One of the new edges should be in the path."
        )

    def test_find_all_rammings_example_set(self, setup_pipe_ramming_example_polygon, debug=False):
        if debug:
            reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)

        osm_graph, mcda_engine, cost_surface_graph = setup_pipe_ramming_example_polygon()

        pipe_ramming = GetPotentialPipeRammingCrossings(osm_graph, cost_surface_graph, debug=debug)
        # Enable for visual checking without full debug mode which slows the test down.
        pipe_ramming.plot_crossings = False
        crossings = pipe_ramming.get_crossings()
        assert len(crossings) > 0


class TestPipeRammingTheoryExamples:
    @pytest.fixture
    def setup_theory_examples(self):
        def _setup(debug: bool = False):
            # Setup clean debug geopackage for plotting.
            if debug:
                out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT
                reset_geopackage(out, truncate=False)
                return out
            return None

        return _setup

    def test_theory_junction_crossing(self, setup_theory_examples):
        pass

    def test_theory_street_segment_crossing(self, setup_theory_examples, debug=False):
        # MCDA vectors
        out = setup_theory_examples(debug=debug)
        street = gpd.GeoDataFrame(
            data=[
                # Asphalt
                [30, shapely.LineString([(0, 0), (100, 0)]).buffer(5, cap_style="flat")],
                # Pavement north
                [5, shapely.LineString([(0, 7), (100, 7)]).buffer(2, cap_style="flat")],
                # Pavement south
                [5, shapely.LineString([(0, -7), (100, -7)]).buffer(2, cap_style="flat")],
            ],
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        buildings = gpd.GeoDataFrame(
            data=[
                [120, shapely.Point(75, -35).buffer(5)],
                [120, shapely.LineString([(10, 15), (45, 15)]).buffer(5, cap_style="flat")],
                [120, shapely.LineString([(5, -15), (45, -15)]).buffer(5, cap_style="flat")],
                [120, shapely.LineString([(55, -15), (95, -15)]).buffer(5, cap_style="flat")],
            ],
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        private_property = gpd.GeoDataFrame(
            data=[
                [70, shapely.LineString([(0, 30), (100, 30)]).buffer(21, cap_style="flat")],
                [70, shapely.LineString([(0, -30), (100, -30)]).buffer(21, cap_style="flat")],
            ],
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        # Note that enabling/disabling these trees changes the selected crossings
        trees = gpd.GeoDataFrame(
            data=[
                [20, shapely.Point(30, -8).buffer(4)],
                [20, shapely.Point(65, 9).buffer(4)],
            ],
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )

        # Create cost-surface
        raster_criteria_groups = {
            "street": "a",
            "buildings": "a",
            "private_property": "a",
            "trees": "b",
        }
        preprocessed_vectors = {
            "street": street,
            "buildings": buildings,
            "private_property": private_property,
            "trees": trees,
        }
        hexagon_graph_builder = HexagonGraphBuilder(
            private_property.union_all(),
            raster_criteria_groups,
            preprocessed_vectors,
            hexagon_size=1,
        )
        cost_surface_graph = hexagon_graph_builder.build_graph()

        # Simple OSM graph, just one street.
        osm_graph = rx.PyGraph()

        node1 = OSMNodeInfo(osm_id=1, geometry=shapely.Point(0, 0))
        node2 = OSMNodeInfo(osm_id=2, geometry=shapely.Point(100, 0))

        node_ids = osm_graph.add_nodes_from([node1, node2])
        node1.node_id, node2.node_id = node_ids

        edges_to_add = [(node1.node_id, node2.node_id, create_edge_info(100, node1, node2))]

        edge_ids = osm_graph.add_edges_from(edges_to_add)
        for edge, edge_id in zip(edges_to_add, edge_ids):
            edge[2].edge_id = edge_id

        pipe_ramming = GetPotentialPipeRammingCrossings(
            osm_graph=osm_graph,
            cost_surface_graph=cost_surface_graph,
            threshold_edge_length_crossing_m=30,
            max_pipe_ramming_length_m=15,
            min_pipe_ramming_length_m=3,
            suitability_value_crossing_threshold=10,
            suitability_value_obstacles_threshold=80,
            hexagon_size=hexagon_graph_builder.hexagon_size,
            debug_out=Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
            debug=debug,
        )
        crossings = pipe_ramming.get_crossings()

        assert len(crossings) == 3
        # Crossings should not intersect any obstacles above the suitability_value_obstacles_threshold
        assert (
            len(
                [
                    i
                    for i in crossings
                    if any(
                        pipe_ramming.cost_surface_nodes[
                            pipe_ramming.cost_surface_nodes.suitability_value
                            > pipe_ramming.suitability_value_obstacles_threshold
                        ].intersects(i[2].geometry.buffer(hexagon_graph_builder.hexagon_size))
                    )
                ]
            )
            == 0
        )

        multi_layer_route_engine = MultilayerRouteEngine(
            pipe_ramming.cost_surface_graph,
            pipe_ramming.osm_graph,
            pipe_ramming.cost_surface_nodes,
            prefix="pytest_theory_",
            write_output=False,
        )
        multi_layer_route_engine.find_route(shapely.LineString([(1, 8), (98, -6)]))

        # Route should cross the street once using a crossing
        crossing_edge_id_pair = [
            (i[0], i[1]) for i in crossings if i[0] and i[1] in multi_layer_route_engine.result_route_node_indices
        ]
        pipe_ramming_edge = pipe_ramming.cost_surface_graph.get_edge_data(
            crossing_edge_id_pair[0][0], crossing_edge_id_pair[0][1]
        )
        assert multi_layer_route_engine.result_route.contains(pipe_ramming_edge.geometry)
        assert multi_layer_route_engine.result_route.length == pytest.approx(123, abs=1)
        assert isinstance(multi_layer_route_engine.result_route, shapely.LineString)

        if debug:
            write_results_to_geopackage(out, street, "pytest_theory_street", overwrite=True)
            write_results_to_geopackage(out, buildings, "pytest_theory_buildings", overwrite=True)
            write_results_to_geopackage(out, private_property, "pytest_theory_private_property", overwrite=True)
            write_results_to_geopackage(out, trees, "pytest_theory_trees", overwrite=True)

            cost_surface_nodes = convert_hexagon_graph_to_gdfs(cost_surface_graph, edges=False)
            write_results_to_geopackage(out, cost_surface_nodes, "pytest_theory_cost_surface_nodes", overwrite=True)
            write_results_to_geopackage(
                out,
                shapely.MultiLineString([i[2].geometry for i in crossings]),
                "pytest_theory_crossings",
                overwrite=True,
            )

            write_results_to_geopackage(
                out, multi_layer_route_engine.result_route, "pytest_theory_result_route", overwrite=True
            )
