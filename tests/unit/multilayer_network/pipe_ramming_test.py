# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pathlib
from enum import Enum, auto

import pandas as pd
import pytest
import geopandas as gpd
import rustworkx as rx
import shapely

from settings import Config
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import convert_hexagon_graph_to_gdfs
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine
from utility_route_planner.util.graph_utilities import create_osm_edge_info
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.multilayer_network.pipe_ramming import GetPotentialPipeRammingCrossings
from utility_route_planner.util.geo_utilities import osm_graph_to_gdfs
from utility_route_planner.models.multilayer_network.osm_graph_preprocessing import OSMGraphPreprocessor
from utility_route_planner.models.multilayer_network.graph_datastructures import OSMNodeInfo
from utility_route_planner.util.write import reset_geopackage, write_results_to_geopackage


class CrossingType(Enum):
    """Helper to control settings for pipe ramming to avoid redundancy in tests."""

    JUNCTION = auto()
    SEGMENT = auto()


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
            (node1.node_id, node2.node_id, create_osm_edge_info(100, node1, node2)),
            (node2.node_id, node3.node_id, create_osm_edge_info(101, node2, node3)),
            (node3.node_id, node4.node_id, create_osm_edge_info(102, node3, node4)),
            (node2.node_id, node5.node_id, create_osm_edge_info(103, node2, node5)),
            (node5.node_id, node6.node_id, create_osm_edge_info(104, node5, node6)),
            (node6.node_id, node7.node_id, create_osm_edge_info(105, node6, node7)),
            (node7.node_id, node8.node_id, create_osm_edge_info(106, node7, node8)),
            (node8.node_id, node9.node_id, create_osm_edge_info(107, node8, node9)),
            (node6.node_id, node9.node_id, create_osm_edge_info(108, node6, node9)),
            (node9.node_id, node10.node_id, create_osm_edge_info(109, node9, node10)),
            (node10.node_id, node11.node_id, create_osm_edge_info(110, node10, node11)),
            (node10.node_id, node12.node_id, create_osm_edge_info(111, node10, node12)),
            (node11.node_id, node12.node_id, create_osm_edge_info(112, node11, node2)),
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
        pipe_ramming.add_crossings_to_graph(crossings)
        assert len(crossings) == 3

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

    @pytest.mark.skip(reason="Longer test for full example set, enable when big (TM) changes are made to pipe ramming.")
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
        def _setup(street, buildings, private_property, trees, debug: bool = False):
            # Assign groups
            raster_criteria_groups = {
                "street": "a",
                "buildings": "a",
                "private_property": "a",
                "trees": "b",
            }
            # Initialize dictionary input for hexagon builder
            preprocessed_vectors = {
                "street": street,
                "buildings": buildings,
                "private_property": private_property,
                "trees": trees,
            }
            # Calculate project area
            project_area = pd.concat([street, buildings, private_property, trees]).union_all()

            if buildings.empty:
                preprocessed_vectors.pop("buildings")
                raster_criteria_groups.pop("buildings")
            if trees.empty:
                preprocessed_vectors.pop("trees")
                raster_criteria_groups.pop("trees")

            hexagon_graph_builder = HexagonGraphBuilder(
                project_area,
                raster_criteria_groups,
                preprocessed_vectors,
                hexagon_size=0.5,
            )
            cost_surface_graph = hexagon_graph_builder.build_graph()

            if debug:
                out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT
                reset_geopackage(out, truncate=False)
            else:
                out = None
            return hexagon_graph_builder, cost_surface_graph, out

        return _setup

    @pytest.mark.parametrize(
        ["buildings", "trees", "n_expected_crossings", "expected_route_length", "start_end"],
        [
            # Without obstacles
            (
                (),
                (),
                3,
                130,
                [(0.3, -6.7), (99, 39.8)],
            ),
            # With obstacles
            (
                [
                    [120, shapely.LineString([(1, -18), (99, -18)]).buffer(8, cap_style="flat")],
                ],
                [
                    [20, shapely.Point(51.5, -7).buffer(4)],
                    [20, shapely.Point(58.5, -7).buffer(4)],
                    [20, shapely.Point(65.5, -7).buffer(4)],
                ],
                2,
                131,
                [(0.3, -6.7), (99, 39.8)],
            ),
        ],
    )
    def test_theory_junction_degree_3_crossing_complex(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length,
        start_end,
        setup_theory_examples,
        debug=True,
    ):
        street = (
            gpd.GeoDataFrame(
                data=[
                    # Street east west
                    [30, shapely.LineString([(0, 0), (100, 0)]).buffer(5, cap_style="flat")],
                    [5, shapely.LineString([(0, 7), (100, 7)]).buffer(2, cap_style="flat")],
                    [5, shapely.LineString([(0, -7), (100, -7)]).buffer(2, cap_style="flat")],
                    # Street north east
                    [5, shapely.LineString([(100, 50), (50, 0)]).buffer(9, cap_style="flat")],
                    [30, shapely.LineString([(100, 50), (50, 0)]).buffer(5, cap_style="flat")],
                ],
                columns=["suitability_value", "geometry"],
                crs=Config.CRS,
            )
            .dissolve(by="suitability_value")
            .explode()
            .reset_index()
        )
        private_property = gpd.GeoDataFrame(
            data=[[70, shapely.Polygon([(0, -50), (100, -50), (100, 50), (0, 50), (0, -50)])]],
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        street = street.clip(private_property.iloc[0].geometry)
        private_property = private_property.overlay(street, how="difference").explode(ignore_index=True)
        buildings = gpd.GeoDataFrame(
            data=buildings,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        hexagon_graph_builder, cost_surface_graph, out = setup_theory_examples(
            street, buildings, private_property, trees, debug=debug
        )

        # OSM graph with a junction
        osm_graph = rx.PyGraph()
        node1 = OSMNodeInfo(osm_id=1, geometry=shapely.Point(0, 0))
        node2 = OSMNodeInfo(osm_id=2, geometry=shapely.Point(50, 0))  # center junction node
        node3 = OSMNodeInfo(osm_id=3, geometry=shapely.Point(100, 50))
        node4 = OSMNodeInfo(osm_id=4, geometry=shapely.Point(100, 0))
        node_ids = osm_graph.add_nodes_from([node1, node2, node3, node4])
        node1.node_id, node2.node_id, node3.node_id, node4.node_id = node_ids
        edges_to_add = [
            (node1.node_id, node2.node_id, create_osm_edge_info(100, node1, node2)),
            (node2.node_id, node3.node_id, create_osm_edge_info(101, node2, node3)),
            (node2.node_id, node4.node_id, create_osm_edge_info(102, node2, node4)),
        ]

        self._run_crossing(
            cost_surface_graph,
            debug,
            edges_to_add,
            expected_route_length,
            hexagon_graph_builder,
            n_expected_crossings,
            osm_graph,
            out,
            start_end,
            CrossingType.JUNCTION,
        )

    @pytest.mark.parametrize(
        ["buildings", "trees", "n_expected_crossings", "expected_route_length", "start_end"],
        [
            # Without obstacles
            (
                (),
                (),
                3,
                106,
                [(0.6, 6.5), (56, 49.5)],
            ),
            # With obstacles
            (
                [
                    [120, shapely.LineString([(1, -18), (99, -18)]).buffer(8, cap_style="flat")],
                    [120, shapely.LineString([(32, 10), (32, 46)]).buffer(8, cap_style="flat")],
                ],
                [[20, shapely.Point(43.6, -8.2).buffer(6)], [20, shapely.Point(58, 19).buffer(4)]],
                2,
                107,
                [(0.6, 6.5), (56, 49.5)],
            ),
        ],
    )
    def test_theory_junction_degree_3_crossing_simple(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length,
        start_end,
        setup_theory_examples,
        debug=False,
    ):
        street = (
            gpd.GeoDataFrame(
                data=[
                    # Street east west
                    [30, shapely.LineString([(0, 0), (100, 0)]).buffer(5, cap_style="flat")],
                    [5, shapely.LineString([(0, 7), (100, 7)]).buffer(2, cap_style="flat")],
                    [5, shapely.LineString([(0, -7), (100, -7)]).buffer(2, cap_style="flat")],
                    # Street north
                    [5, shapely.LineString([(43, 50), (43, 0)]).buffer(2, cap_style="flat")],
                    [30, shapely.LineString([(50, 50), (50, 0)]).buffer(5, cap_style="flat")],
                    [5, shapely.LineString([(57, 50), (57, 0)]).buffer(2, cap_style="flat")],
                ],
                columns=["suitability_value", "geometry"],
                crs=Config.CRS,
            )
            .dissolve(by="suitability_value")
            .explode()
            .reset_index()
        )
        private_property = gpd.GeoDataFrame(
            data=[[70, shapely.Polygon([(0, -50), (100, -50), (100, 50), (0, 50), (0, -50)])]],
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        private_property = private_property.overlay(street, how="difference").explode(ignore_index=True)
        buildings = gpd.GeoDataFrame(
            data=buildings,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        hexagon_graph_builder, cost_surface_graph, out = setup_theory_examples(
            street, buildings, private_property, trees, debug=debug
        )

        # OSM graph with a junction
        osm_graph = rx.PyGraph()
        node1 = OSMNodeInfo(osm_id=1, geometry=shapely.Point(0, 0))
        node2 = OSMNodeInfo(osm_id=2, geometry=shapely.Point(50, 0))
        node3 = OSMNodeInfo(osm_id=3, geometry=shapely.Point(100, 0))
        node4 = OSMNodeInfo(osm_id=4, geometry=shapely.Point(50, 50))
        node_ids = osm_graph.add_nodes_from([node1, node2, node3, node4])
        node1.node_id, node2.node_id, node3.node_id, node4.node_id = node_ids
        edges_to_add = [
            (node1.node_id, node2.node_id, create_osm_edge_info(100, node1, node2)),
            (node2.node_id, node3.node_id, create_osm_edge_info(101, node2, node3)),
            (node2.node_id, node4.node_id, create_osm_edge_info(102, node2, node4)),
        ]

        self._run_crossing(
            cost_surface_graph,
            debug,
            edges_to_add,
            expected_route_length,
            hexagon_graph_builder,
            n_expected_crossings,
            osm_graph,
            out,
            start_end,
            CrossingType.JUNCTION,
        )

    @pytest.mark.parametrize(
        ["buildings", "trees", "n_expected_crossings", "expected_route_length", "start_end"],
        [
            # Without obstacles
            (
                (),
                (),
                4,
                118,
                [(0.3, -6.7), (56.23, 49.68)],
            ),
            # With obstacles
            (
                [
                    # south west
                    [120, shapely.LineString([(1, -18), (40, -18)]).buffer(8, cap_style="flat")],
                    [120, shapely.LineString([(32, -18), (32, -49)]).buffer(8, cap_style="flat")],
                    # north east
                    [120, shapely.LineString([(1, 18), (40, 18)]).buffer(8, cap_style="flat")],
                    [120, shapely.LineString([(32, 18), (32, 49)]).buffer(8, cap_style="flat")],
                    # north west
                    [120, shapely.LineString([(68, 49), (68, 18)]).buffer(8, cap_style="flat")],
                    [120, shapely.LineString([(60, 18), (99, 18)]).buffer(8, cap_style="flat")],
                    # south east
                    [120, shapely.LineString([(68, -49), (68, -18)]).buffer(8, cap_style="flat")],
                    [120, shapely.LineString([(60, -18), (99, -18)]).buffer(8, cap_style="flat")],
                ],
                [
                    [20, shapely.Point(42.573, 8.294).buffer(4)],
                ],
                4,
                118,
                [(0.3, -6.7), (56.23, 49.68)],
            ),
        ],
    )
    def test_theory_junction_degree_4_crossing_simple(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length,
        start_end,
        setup_theory_examples,
        debug=False,
    ):
        street = (
            gpd.GeoDataFrame(
                data=[
                    # Street east west
                    [30, shapely.LineString([(0, 0), (100, 0)]).buffer(5, cap_style="flat")],
                    [5, shapely.LineString([(0, 7), (100, 7)]).buffer(2, cap_style="flat")],
                    [5, shapely.LineString([(0, -7), (100, -7)]).buffer(2, cap_style="flat")],
                    # Street north south
                    [5, shapely.LineString([(50, -50), (50, 50)]).buffer(9, cap_style="flat")],
                    [30, shapely.LineString([(50, -50), (50, 50)]).buffer(5, cap_style="flat")],
                ],
                columns=["suitability_value", "geometry"],
                crs=Config.CRS,
            )
            .dissolve(by="suitability_value")
            .explode()
            .reset_index()
        )
        private_property = gpd.GeoDataFrame(
            data=[[70, shapely.Polygon([(0, -50), (100, -50), (100, 50), (0, 50), (0, -50)])]],
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        street = street.clip(private_property.iloc[0].geometry)
        private_property = private_property.overlay(street, how="difference").explode(ignore_index=True)
        buildings = (
            gpd.GeoDataFrame(
                data=buildings,
                columns=["suitability_value", "geometry"],
                crs=Config.CRS,
            )
            .dissolve(by="suitability_value")
            .explode()
            .reset_index()
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        hexagon_graph_builder, cost_surface_graph, out = setup_theory_examples(
            street, buildings, private_property, trees, debug=debug
        )

        # OSM graph with a junction
        osm_graph = rx.PyGraph()
        node1 = OSMNodeInfo(osm_id=1, geometry=shapely.Point(0, 0))
        node2 = OSMNodeInfo(osm_id=2, geometry=shapely.Point(50, 0))  # center junction node
        node3 = OSMNodeInfo(osm_id=3, geometry=shapely.Point(50, 50))
        node4 = OSMNodeInfo(osm_id=4, geometry=shapely.Point(100, 0))
        node5 = OSMNodeInfo(osm_id=5, geometry=shapely.Point(50, -50))
        node_ids = osm_graph.add_nodes_from([node1, node2, node3, node4, node5])
        node1.node_id, node2.node_id, node3.node_id, node4.node_id, node5.node_id = node_ids
        edges_to_add = [
            (node1.node_id, node2.node_id, create_osm_edge_info(100, node1, node2)),
            (node2.node_id, node3.node_id, create_osm_edge_info(101, node2, node3)),
            (node2.node_id, node4.node_id, create_osm_edge_info(102, node2, node4)),
            (node2.node_id, node5.node_id, create_osm_edge_info(103, node2, node5)),
        ]

        self._run_crossing(
            cost_surface_graph,
            debug,
            edges_to_add,
            expected_route_length,
            hexagon_graph_builder,
            n_expected_crossings,
            osm_graph,
            out,
            start_end,
            CrossingType.JUNCTION,
        )

    @pytest.mark.parametrize(
        ["buildings", "trees", "n_expected_crossings", "expected_route_length", "start_end"],
        [
            # Without obstacles
            (
                (),
                (),
                10,
                124,
                ((1, -3), (93, 45)),
            ),
            # With obstacles
            (
                [[120, shapely.LineString([(2, -26), (44, -18), (63, -36)]).buffer(8, cap_style="flat")]],
                [[20, shapely.Point(43, 6).buffer(4)]],
                8,
                138,
                [(1, -3), (93, 45)],
            ),
        ],
    )
    def test_theory_junction_degree_4_crossing_complex(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length,
        start_end,
        setup_theory_examples,
        debug=False,
    ):
        street = (
            gpd.GeoDataFrame(
                data=[
                    # Street north
                    [30, shapely.LineString([(50, 50), (50, 0)]).buffer(5, cap_style="flat")],
                    [5, shapely.LineString([(50, 50), (50, 0)]).buffer(7, cap_style="flat")],
                    # Street north-east
                    [30, shapely.LineString([(50, 0), (90, 50)]).buffer(5, cap_style="flat")],
                    [5, shapely.LineString([(50, 0), (90, 50)]).buffer(7, cap_style="flat")],
                    # Street south-west
                    [30, shapely.LineString([(50, 0), (100, -50)]).buffer(5, cap_style="flat")],
                    [5, shapely.LineString([(50, 0), (100, -50)]).buffer(7, cap_style="flat")],
                    # Street west
                    [30, shapely.LineString([(50, 0), (0, -10)]).buffer(5, cap_style="flat")],
                    [5, shapely.LineString([(50, 0), (0, -10)]).buffer(7, cap_style="flat")],
                ],
                columns=["suitability_value", "geometry"],
                crs=Config.CRS,
            )
            .dissolve(by="suitability_value")
            .explode()
            .reset_index()
        )
        private_property = gpd.GeoDataFrame(
            data=[[70, shapely.Polygon([(0, -50), (100, -50), (100, 50), (0, 50), (0, -50)])]],
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        street = street.clip(private_property.iloc[0].geometry)
        private_property = private_property.overlay(street, how="difference").explode(ignore_index=True)
        buildings = gpd.GeoDataFrame(
            data=buildings,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        hexagon_graph_builder, cost_surface_graph, out = setup_theory_examples(
            street, buildings, private_property, trees, debug=debug
        )

        # OSM graph with a junction
        osm_graph = rx.PyGraph()
        node1 = OSMNodeInfo(osm_id=1, geometry=shapely.Point(50, 0))  # center junction node
        node2 = OSMNodeInfo(osm_id=2, geometry=shapely.Point(50, 50))
        node3 = OSMNodeInfo(osm_id=3, geometry=shapely.Point(90, 50))
        node4 = OSMNodeInfo(osm_id=4, geometry=shapely.Point(100, -50))
        node5 = OSMNodeInfo(osm_id=5, geometry=shapely.Point(0, -10))
        node_ids = osm_graph.add_nodes_from([node1, node2, node3, node4, node5])
        node1.node_id, node2.node_id, node3.node_id, node4.node_id, node5.node_id = node_ids
        edges_to_add = [
            (node1.node_id, node2.node_id, create_osm_edge_info(105, node1, node2)),
            (node1.node_id, node3.node_id, create_osm_edge_info(106, node1, node3)),
            (node1.node_id, node4.node_id, create_osm_edge_info(107, node1, node4)),
            (node1.node_id, node5.node_id, create_osm_edge_info(108, node1, node5)),
        ]

        self._run_crossing(
            cost_surface_graph,
            debug,
            edges_to_add,
            expected_route_length,
            hexagon_graph_builder,
            n_expected_crossings,
            osm_graph,
            out,
            start_end,
            CrossingType.JUNCTION,
        )

    @pytest.mark.parametrize(
        ["buildings", "trees", "n_expected_crossings", "expected_route_length", "start_end"],
        [
            # Without obstacles
            (
                (),
                (),
                3,
                123,
                [(1, 8), (98, -6)],
            ),
            # With obstacles
            (
                [
                    [120, shapely.LineString([(10, 15), (45, 15)]).buffer(5, cap_style="flat")],
                    [120, shapely.LineString([(5, -15), (45, -15)]).buffer(5, cap_style="flat")],
                    [120, shapely.LineString([(55, -15), (95, -15)]).buffer(5, cap_style="flat")],
                ],
                [[20, shapely.Point(30, -8).buffer(4)], [20, shapely.Point(65, 9).buffer(4)]],
                3,
                123,
                [(1, 8), (98, -6)],
            ),
        ],
    )
    def test_theory_segment_crossing_straight_street(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length,
        start_end,
        setup_theory_examples,
        debug=False,
    ):
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
        private_property = gpd.GeoDataFrame(
            data=[
                [70, shapely.LineString([(0, 30), (100, 30)]).buffer(21, cap_style="flat")],
                [70, shapely.LineString([(0, -30), (100, -30)]).buffer(21, cap_style="flat")],
            ],
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        buildings = gpd.GeoDataFrame(
            data=buildings,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        # MCDA vectors
        hexagon_graph_builder, cost_surface_graph, out = setup_theory_examples(
            street, buildings, private_property, trees, debug=debug
        )

        # Create a simple OSM graph, just one street.
        osm_graph = rx.PyGraph()
        node1 = OSMNodeInfo(osm_id=1, geometry=shapely.Point(0, 0))
        node2 = OSMNodeInfo(osm_id=2, geometry=shapely.Point(100, 0))
        node_ids = osm_graph.add_nodes_from([node1, node2])
        node1.node_id, node2.node_id = node_ids
        edges_to_add = [(node1.node_id, node2.node_id, create_osm_edge_info(100, node1, node2))]

        self._run_crossing(
            cost_surface_graph,
            debug,
            edges_to_add,
            expected_route_length,
            hexagon_graph_builder,
            n_expected_crossings,
            osm_graph,
            out,
            start_end,
            CrossingType.SEGMENT,
        )

    @pytest.mark.parametrize(
        ["buildings", "trees", "n_expected_crossings", "expected_route_length", "start_end"],
        [
            # Without obstacles
            (
                (),
                (),
                6,
                233,
                [(1, 6), (158, -25)],
            ),
            # With obstacles
            (
                [
                    [
                        120,
                        shapely.LineString([(55.4, 33.4), (80.5, -32)])
                        .buffer(11, cap_style="flat", single_sided=True)
                        .buffer(-1, cap_style="flat"),
                    ],
                    [
                        120,
                        shapely.LineString([(69.4, -47.7), (48.3, 7.1)])
                        .buffer(11, cap_style="flat", single_sided=True)
                        .buffer(-1, cap_style="flat"),
                    ],
                    [
                        120,
                        shapely.LineString([(17, 8), (53.5, 38.4)])
                        .buffer(11, cap_style="flat", single_sided=True)
                        .buffer(-1, cap_style="flat"),
                    ],
                ],
                [
                    [20, shapely.Point(30, -8).buffer(4)],
                    [20, shapely.Point(39.804, 6.896).buffer(4)],
                    [20, shapely.Point(47.260, 10.421).buffer(6)],
                    [20, shapely.Point(87, -30.9).buffer(4)],
                    [20, shapely.Point(75.1, -47.6).buffer(6.5)],
                ],
                4,
                245,
                [(1, 6), (158, -25)],
            ),
        ],
    )
    def test_theory_segment_crossing_complex_street(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length,
        start_end,
        setup_theory_examples,
        debug=False,
    ):
        street_linestring = shapely.LineString(
            [(0, 0), (20, 0), (50, 25), (75, -40), (120, -40), (150, -20), (160, -20)]
        )
        street = (
            gpd.GeoDataFrame(
                data=[
                    [30, street_linestring.buffer(5, cap_style="flat")],
                    [
                        5,
                        shapely.difference(
                            street_linestring.buffer(8, cap_style="flat"),
                            street_linestring.buffer(5, cap_style="flat"),
                            grid_size=0.01,
                        ),
                    ],
                ],
                columns=["suitability_value", "geometry"],
                crs=Config.CRS,
            )
            .explode()
            .reset_index(drop=True)
        )
        private_property = gpd.GeoDataFrame(
            data=[
                [70, shapely.LineString([(0, 0), (160, 0)]).buffer(90, cap_style="flat")],
            ],
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        private_property = private_property.overlay(street, how="difference").explode(ignore_index=True)
        buildings = gpd.GeoDataFrame(
            data=buildings,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "geometry"],
            crs=Config.CRS,
        )
        # MCDA vectors
        hexagon_graph_builder, cost_surface_graph, out = setup_theory_examples(
            street, buildings, private_property, trees, debug=debug
        )

        # Create a simple OSM graph, just one street.
        osm_graph = rx.PyGraph()
        node1 = OSMNodeInfo(osm_id=1, geometry=shapely.Point(0, 0))
        node2 = OSMNodeInfo(osm_id=2, geometry=shapely.Point(20, 0))
        node3 = OSMNodeInfo(osm_id=3, geometry=shapely.Point(50, 25))
        node4 = OSMNodeInfo(osm_id=4, geometry=shapely.Point(75, -40))
        node5 = OSMNodeInfo(osm_id=5, geometry=shapely.Point(120, -40))
        node6 = OSMNodeInfo(osm_id=6, geometry=shapely.Point(150, -20))
        node7 = OSMNodeInfo(osm_id=7, geometry=shapely.Point(160, -20))
        node_ids = osm_graph.add_nodes_from([node1, node2, node3, node4, node5, node6, node7])
        node1.node_id, node2.node_id, node3.node_id, node4.node_id, node5.node_id, node6.node_id, node7.node_id = (
            node_ids
        )
        edges_to_add = [
            (node1.node_id, node2.node_id, create_osm_edge_info(100, node1, node2)),
            (node2.node_id, node3.node_id, create_osm_edge_info(101, node2, node3)),
            (node3.node_id, node4.node_id, create_osm_edge_info(102, node3, node4)),
            (node4.node_id, node5.node_id, create_osm_edge_info(103, node4, node5)),
            (node5.node_id, node6.node_id, create_osm_edge_info(104, node5, node6)),
            (node6.node_id, node7.node_id, create_osm_edge_info(105, node6, node7)),
        ]

        self._run_crossing(
            cost_surface_graph,
            debug,
            edges_to_add,
            expected_route_length,
            hexagon_graph_builder,
            n_expected_crossings,
            osm_graph,
            out,
            start_end,
            CrossingType.SEGMENT,
        )

    def test_theory_scenario_with_bridge(self):
        pass

    def _run_crossing(
        self,
        cost_surface_graph: rx.PyGraph,
        debug: bool,
        edges_to_add: list,
        expected_route_length: float,
        hexagon_graph_builder: HexagonGraphBuilder,
        n_expected_crossings: int,
        osm_graph: rx.PyGraph,
        out: pathlib.Path,
        start_end: list[tuple],
        crossing_type: CrossingType,
    ):
        match crossing_type:
            case crossing_type.JUNCTION:
                threshold_edge_length_crossing_m = 100
            case crossing_type.SEGMENT:
                threshold_edge_length_crossing_m = 30
            case _:
                raise ValueError("Invalid crossing type specified.")

        edge_ids = osm_graph.add_edges_from(edges_to_add)
        for edge, edge_id in zip(edges_to_add, edge_ids):
            edge[2].edge_id = edge_id
        pipe_ramming = GetPotentialPipeRammingCrossings(
            osm_graph=osm_graph,
            cost_surface_graph=cost_surface_graph,
            threshold_edge_length_crossing_m=threshold_edge_length_crossing_m,
            max_pipe_ramming_length_m=20,
            min_pipe_ramming_length_m=3,
            suitability_value_crossing_threshold=10,
            suitability_value_obstacles_threshold=80,
            hexagon_size=hexagon_graph_builder.hexagon_size,
            debug_out=Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT,
            debug=debug,
        )
        crossings = pipe_ramming.get_crossings()
        self._assert_crossings(crossings, hexagon_graph_builder, pipe_ramming, n_expected_crossings)
        multilayer_route_engine = MultilayerRouteEngine(
            pipe_ramming.cost_surface_graph,
            pipe_ramming.osm_graph,
            pipe_ramming.cost_surface_nodes,
            prefix="pytest_theory_",
            write_output=False,
        )
        multilayer_route_engine.find_route(shapely.LineString(start_end))
        assert multilayer_route_engine.result_route.length == pytest.approx(expected_route_length, abs=1)
        self._assert_result_route(crossings, multilayer_route_engine, pipe_ramming)
        if debug:
            self._plot_pytest_theory(
                out,
                cost_surface_graph,
                crossings,
                multilayer_route_engine,
                pipe_ramming,
                hexagon_graph_builder.preprocessed_vectors,
            )

    @staticmethod
    def _assert_crossings(
        crossings: list,
        hexagon_graph_builder: HexagonGraphBuilder,
        pipe_ramming: GetPotentialPipeRammingCrossings,
        n_expected_crossings: int,
    ):
        assert len(crossings) == n_expected_crossings
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

    @staticmethod
    def _assert_result_route(
        crossings: list, multilayer_route_engine: MultilayerRouteEngine, pipe_ramming: GetPotentialPipeRammingCrossings
    ):
        crossing_edge_id_pairs = [
            (i[0], i[1])
            for i in crossings
            if i[0] and i[1] in multilayer_route_engine.result_route_node_indices  # type: ignore
        ]
        pipe_ramming_edges = [
            pipe_ramming.cost_surface_graph.get_edge_data(i[0], i[1]).geometry for i in crossing_edge_id_pairs
        ]
        assert multilayer_route_engine.result_route.intersects(shapely.MultiLineString(pipe_ramming_edges))
        assert isinstance(multilayer_route_engine.result_route, shapely.LineString)

    @staticmethod
    def _plot_pytest_theory(
        out: pathlib.Path,
        cost_surface_graph: rx.PyGraph,
        crossings: list,
        multilayer_route_engine: MultilayerRouteEngine,
        pipe_ramming: GetPotentialPipeRammingCrossings,
        preprocessed_vectors: dict,
    ):
        # MCDA vectors
        write_results_to_geopackage(out, preprocessed_vectors["street"], "pytest_theory_street", overwrite=True)
        write_results_to_geopackage(
            out, preprocessed_vectors["private_property"], "pytest_theory_private_property", overwrite=True
        )
        if "buildings" in preprocessed_vectors:
            write_results_to_geopackage(
                out, preprocessed_vectors["buildings"], "pytest_theory_buildings", overwrite=True
            )
        if "trees" in preprocessed_vectors:
            write_results_to_geopackage(out, preprocessed_vectors["trees"], "pytest_theory_trees", overwrite=True)
        # OSM graph
        write_results_to_geopackage(out, pipe_ramming.osm_nodes, "pytest_theory_osm_nodes", overwrite=True)
        write_results_to_geopackage(out, pipe_ramming.osm_edges, "pytest_theory_osm_edges", overwrite=True)
        # Cost-surface & crossings
        cost_surface_nodes, cost_surface_edges = convert_hexagon_graph_to_gdfs(cost_surface_graph, edges=True)
        write_results_to_geopackage(out, cost_surface_nodes, "pytest_theory_cost_surface_nodes", overwrite=True)
        write_results_to_geopackage(out, cost_surface_edges, "pytest_theory_cost_surface_edges", overwrite=True)
        write_results_to_geopackage(
            out,
            shapely.MultiLineString([i[2].geometry for i in crossings]),
            "pytest_theory_crossings",
            overwrite=True,
        )
        # Resulting route
        write_results_to_geopackage(
            out, multilayer_route_engine.result_route, "pytest_theory_result_route", overwrite=True
        )


class TestPipeRammingUtils:
    def test_validate_node_pairs_merges_multi_groups(self):
        # Create a MultiIndex Series with more than 2 entries for one group
        idx = pd.MultiIndex.from_tuples(
            [
                (1, "a"),
                (1, "b"),
                (1, "c"),  # group 1 has 3 entries
                (2, "a"),
                (2, "b"),  # group 2 has 2 entries
            ],
            names=["index_right", "idx_street_side"],
        )
        s = pd.Series([10, 20, 30, 40, 50], index=idx)

        result = GetPotentialPipeRammingCrossings._validate_node_pairs(s)

        # Group 1 should be split into new pairs, group 2 should remain unchanged
        assert isinstance(result, pd.Series)
        assert set(result.index.get_level_values(0)) >= {2, 3, 4}  # Group 1 is replaced with 3 and 4
        # Check that all original nodes are present
        assert set(result.values) >= {10, 20, 30, 40, 50}

        idx_expected = pd.MultiIndex.from_tuples(
            [
                (2, "a"),
                (2, "b"),
                (3, "a"),
                (3, "b"),
                (4, "a"),
                (4, "c"),
            ],
            names=["index_right", "idx_street_side"],
        )
        expected = pd.Series([40, 50, 10, 20, 10, 30], index=idx_expected)
        pd.testing.assert_series_equal(result, expected)

    def test_validate_node_pairs_no_multi_groups(self):
        # Only groups with <=2 entries
        idx = pd.MultiIndex.from_tuples(
            [
                (1, "a"),
                (1, "b"),
                (2, "a"),
                (2, "b"),
            ],
            names=["index_right", "idx_street_side"],
        )
        s = pd.Series([10, 20, 30, 40], index=idx)

        result = GetPotentialPipeRammingCrossings._validate_node_pairs(s)
        # Should return the input unchanged
        pd.testing.assert_series_equal(result, s)
