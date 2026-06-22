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
from tests.integration.conftest import write_criteria_vectors
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_edge_generator import HexagonEdgeGenerator
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_composer import build_and_compose_graph
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_grid_builder import HexagonGridBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import (
    convert_hexagon_edges_to_gdf,
)
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine
from utility_route_planner.util.graph_utilities import build_osm_test_graph
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.multilayer_network.pipe_ramming import (
    GetPotentialPipeRammingCrossings,
    PipeRammingSettings,
)
from utility_route_planner.util.geo_utilities import osm_graph_to_gdfs
from utility_route_planner.models.multilayer_network.osm_graph_preprocessing import OSMGraphPreprocessor
from utility_route_planner.util.write import reset_geopackage, write_results_to_geopackage


class CrossingType(Enum):
    """Helper to control settings for pipe ramming to avoid redundancy in tests."""

    JUNCTION = auto()
    SEGMENT = auto()


class TestPipeRamming:
    debug: bool = Config.DEBUG
    hexagon_size: float = 0.5
    out: pathlib.Path = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT

    @pytest.fixture
    def setup_pipe_ramming_example_for_ede_polygon(self, load_osm_graph_pickle):
        def _setup(project_area=None):
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

            grid_builder = HexagonGridBuilder(hexagon_size=Config.HEXAGON_SIZE, block_size=Config.HEXAGON_BLOCK_SIZE)
            hexagon_edge_generator = HexagonEdgeGenerator()
            hexagon_graph_builder = HexagonGraphBuilder(
                grid_builder=grid_builder, edge_generator=hexagon_edge_generator
            )

            raster_groups = {
                criteria_key: criteria.group for criteria_key, criteria in mcda_engine.raster_preset.criteria.items()
            }
            cost_surface_graph, cost_surface_nodes = hexagon_graph_builder.build_graph(
                project_area, raster_groups, mcda_engine.processed_vectors
            )

            if self.debug:
                osm_nodes, osm_edges = osm_graph_to_gdfs(osm_graph_preprocessed)
                reset_geopackage(self.out, truncate=False)
                write_results_to_geopackage(self.out, osm_nodes, "osm_nodes")
                write_results_to_geopackage(self.out, osm_edges, "osm_edges")
                write_results_to_geopackage(self.out, cost_surface_nodes, "cost_surface_nodes")

            return osm_graph_preprocessed, mcda_engine, cost_surface_graph, cost_surface_nodes

        return _setup

    def test_create_road_segment_groups(self):
        if self.debug:
            reset_geopackage(self.out, truncate=False)

        osm_graph = build_osm_test_graph(
            nodes=[
                (1, (0, 0)),
                (2, (1, 0)),
                (3, (1, -1)),
                (4, (1, -2)),
                (5, (2, 0)),
                (6, (3, 0)),
                (7, (3, 1)),
                (8, (4, 1)),
                (9, (4, 0)),
                (10, (5, 0)),
                (11, (6, 1)),
                (12, (6, -1)),
            ],
            edges=[
                (1, 2, 100),
                (2, 3, 101),
                (3, 4, 102),
                (2, 5, 103),
                (5, 6, 104),
                (6, 7, 105),
                (7, 8, 106),
                (8, 9, 107),
                (6, 9, 108),
                (9, 10, 109),
                (10, 11, 110),
                (10, 12, 111),
                (11, 12, 112),
            ],
        )

        # Enable debug for visual debugging in QGIS.
        crossings = GetPotentialPipeRammingCrossings(
            osm_graph, cost_surface_graph=rx.PyGraph(), cost_surface_nodes=gpd.GeoDataFrame(), debug=self.debug
        )
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

    @pytest.mark.skip(reason="Only for debugging a specific junction.")
    def test_single_junction(self, setup_pipe_ramming_example_for_ede_polygon):
        if self.debug:
            reset_geopackage(self.out, truncate=False)

        node_id_to_test = 3
        project_area = shapely.Point(174967.12, 450898.60).buffer(150)

        osm_graph, _, cost_surface_graph, cost_surface_nodes = setup_pipe_ramming_example_for_ede_polygon(project_area)

        max_pipe_ramming_length_m = 27  # play with value and note that crossings move
        pipe_ramming = GetPotentialPipeRammingCrossings(
            osm_graph,
            cost_surface_graph,
            cost_surface_nodes,
            settings=PipeRammingSettings(max_pipe_ramming_length_m=max_pipe_ramming_length_m),
            debug=self.debug,
        )
        pipe_ramming.create_road_segment_groups()
        pipe_ramming.prepare_junction_crossings()
        crossings = pipe_ramming.get_crossing_for_junction(
            node_id_to_test,
            pipe_ramming.junctions_of_interests.loc[node_id_to_test].osm_id,
            pipe_ramming.junctions_of_interests.loc[node_id_to_test].geometry,
            pipe_ramming.junctions_of_interests.loc[node_id_to_test].degree,
        )
        pipe_ramming.add_crossings_to_graph(crossings)

        # Test our newly found crossing in a shortest path.
        pipe_ramming.add_crossings_to_graph(crossings)
        start_end = shapely.LineString([(174971.62, 450911.846), (174971.62, 450888.463)])
        multilayer_route_engine = MultilayerRouteEngine(
            pipe_ramming.cost_surface_graph,
            pipe_ramming.osm_graph,
            pipe_ramming.cost_surface_nodes,
            hexagon_size=self.hexagon_size,
            prefix="pytest_junction_",
        )
        multilayer_route_engine.find_route(start_end)

        assert len(crossings) == 3
        assert multilayer_route_engine.get_result_route_length_unprocessed() == pytest.approx(25, abs=1)
        assert len([i for i in crossings if i[0] and i[1] in multilayer_route_engine.results.node_indices]) == 1, (
            "One of the new edges should be in the path."
        )

    @pytest.mark.skip(reason="Only for debugging a specific road-segment group.")
    def test_single_road_segment_group(self, setup_pipe_ramming_example_for_ede_polygon):
        if self.debug:
            reset_geopackage(self.out, truncate=False)

        segment_group_to_cross = 48

        project_area = shapely.Point(174974, 451093).buffer(150)
        osm_graph, _, cost_surface_graph, cost_surface_nodes = setup_pipe_ramming_example_for_ede_polygon(project_area)

        pipe_ramming = GetPotentialPipeRammingCrossings(
            osm_graph, cost_surface_graph, cost_surface_nodes, debug=self.debug
        )
        pipe_ramming.suitability_value_obstacles_threshold = 77
        pipe_ramming.create_road_segment_groups()
        pipe_ramming.prepare_junction_crossings()
        segments_of_interest = pipe_ramming.prepare_segment_crossings()
        crossings = pipe_ramming.get_crossings_per_segment(
            segment_group_to_cross, segments_of_interest.loc[segment_group_to_cross].geometry
        )
        pipe_ramming.add_crossings_to_graph(crossings)

        start_end = shapely.LineString([(174927.5, 451098.452), (174932, 451089.791)])
        multilayer_route_engine = MultilayerRouteEngine(
            pipe_ramming.cost_surface_graph,
            pipe_ramming.osm_graph,
            pipe_ramming.cost_surface_nodes,
            hexagon_size=self.hexagon_size,
            prefix="pytest_junction_",
        )
        multilayer_route_engine.find_route(start_end)

        assert len(crossings) == 3
        assert multilayer_route_engine.get_result_route_length_unprocessed() == pytest.approx(12, abs=1)
        assert len([i for i in crossings if i[0] and i[1] in multilayer_route_engine.results.node_indices]) == 1, (
            "One of the new edges should be in the path."
        )

    @pytest.mark.skip(reason="Longer test for full example set, enable when big (TM) changes are made to pipe ramming.")
    def test_find_all_rammings_example_set(self, setup_pipe_ramming_example_for_ede_polygon):
        if self.debug:
            reset_geopackage(self.out, truncate=False)

        osm_graph, _, cost_surface_graph, cost_surface_nodes = setup_pipe_ramming_example_for_ede_polygon()

        pipe_ramming = GetPotentialPipeRammingCrossings(
            osm_graph, cost_surface_graph, cost_surface_nodes, debug=self.debug
        )
        # Enable for visual checking without full debug mode which slows the test down.
        pipe_ramming.plot_crossings = False
        crossings = pipe_ramming.get_crossings()
        assert len(crossings) > 0


class TestPipeRammingTheoryExamples:
    debug: bool = False
    out: pathlib.Path = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT
    hexagon_size = 0.5
    block_size = 32
    prefix: str = "pytest_"

    @pytest.fixture
    def setup_theory_examples(self):
        def _setup(road, buildings, private_property, trees, osm_graph, crossing_type: CrossingType):
            if self.debug:
                reset_geopackage(self.out, truncate=False)

            # Assign groups
            raster_criteria_groups = {
                "road": "a",
                "buildings": "a",
                "private_property": "a",
                "trees": "b",
            }
            # Initialize dictionary input for hexagon builder
            preprocessed_vectors = {
                "road": road,
                "buildings": buildings,
                "private_property": private_property,
                "trees": trees,
            }
            # Calculate project area, take a tiny buffer to avoid intersecting problems with geopandas.
            project_area = pd.concat([road, buildings, private_property, trees]).union_all().buffer(0.01)

            processed_criteria_per_height_level = {0: ["road", "buildings", "private_property", "trees"]}

            if buildings.empty:
                preprocessed_vectors.pop("buildings")
                raster_criteria_groups.pop("buildings")
                processed_criteria_per_height_level[0].remove("buildings")
            if trees.empty:
                preprocessed_vectors.pop("trees")
                raster_criteria_groups.pop("trees")
                processed_criteria_per_height_level[0].remove("trees")

            match crossing_type:
                case crossing_type.JUNCTION:
                    threshold_edge_length_crossing_m = 100
                case crossing_type.SEGMENT:
                    threshold_edge_length_crossing_m = 30
                case _:
                    raise ValueError("Invalid crossing type specified.")

            settings = PipeRammingSettings(
                hexagon_size=self.hexagon_size,
                threshold_edge_length_crossing_m=threshold_edge_length_crossing_m,
                max_pipe_ramming_length_m=20,
                min_pipe_ramming_length_m=3,
                suitability_value_crossing_threshold=10,
                suitability_value_obstacles_threshold=80,
            )

            cost_surface_graph, cost_surface_nodes, crossings = build_and_compose_graph(
                processed_criteria_per_height_level=processed_criteria_per_height_level,
                processed_criteria_vectors=preprocessed_vectors,
                raster_groups=raster_criteria_groups,
                project_area=project_area,
                debug=self.debug,
                hexagon_size=self.hexagon_size,
                block_size=self.block_size,
                apply_pipe_ramming=True,
                osm_graph_preprocessed=osm_graph,
                pipe_ramming_settings=settings,
            )

            if self.debug:
                write_criteria_vectors(project_area, preprocessed_vectors)
                nodes, edges = osm_graph_to_gdfs(osm_graph)
                write_results_to_geopackage(self.out, nodes, f"{self.prefix}osm_nodes", overwrite=True)
                write_results_to_geopackage(self.out, edges, f"{self.prefix}osm_edges", overwrite=True)
                cost_surface_edges = convert_hexagon_edges_to_gdf(cost_surface_graph, cost_surface_nodes)
                write_results_to_geopackage(
                    self.out, cost_surface_nodes, f"{self.prefix}cost_surface_nodes", overwrite=True
                )
                write_results_to_geopackage(
                    self.out, cost_surface_edges, f"{self.prefix}cost_surface_edges", overwrite=True
                )

            return cost_surface_graph, cost_surface_nodes, crossings, settings

        return _setup

    @pytest.mark.parametrize(
        [
            "buildings",
            "trees",
            "n_expected_crossings",
            "expected_route_length_unprocessed",
            "expected_route_length_straightened",
            "expected_route_cost",
            "expected_crossings_used_in_route",
            "start_end",
        ],
        [
            # Without obstacles
            (
                (),
                (),
                3,
                130,
                116,
                805,
                1,
                [(0.3, -6.7), (99, 39.8)],
            ),
            # With obstacles
            (
                [
                    [120, 0, shapely.LineString([(1, -18), (99, -18)]).buffer(8, cap_style="flat")],
                ],
                [
                    [20, 0, shapely.Point(51.5, -7).buffer(4)],
                    [20, 0, shapely.Point(58.5, -7).buffer(4)],
                    [20, 0, shapely.Point(65.5, -7).buffer(4)],
                ],
                2,
                128,
                117,
                840,
                2,
                [(0.3, -6.7), (99, 39.8)],
            ),
        ],
    )
    def test_theory_junction_degree_3_crossing_complex(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length_unprocessed,
        expected_route_length_straightened,
        expected_route_cost,
        expected_crossings_used_in_route,
        start_end,
        setup_theory_examples,
    ):
        road = (
            gpd.GeoDataFrame(
                data=[
                    # road east west
                    [30, 0, shapely.LineString([(0, 0), (100, 0)]).buffer(5, cap_style="flat")],
                    [5, 0, shapely.LineString([(0, 7), (100, 7)]).buffer(2, cap_style="flat")],
                    [5, 0, shapely.LineString([(0, -7), (100, -7)]).buffer(2, cap_style="flat")],
                    # road north east
                    [5, 0, shapely.LineString([(100, 50), (50, 0)]).buffer(9, cap_style="flat")],
                    [30, 0, shapely.LineString([(100, 50), (50, 0)]).buffer(5, cap_style="flat")],
                ],
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
                crs=Config.CRS,
            )
            .dissolve(by="suitability_value")
            .explode()
            .reset_index()
        )
        private_property = gpd.GeoDataFrame(
            data=[[70, 0, shapely.Polygon([(0, -50), (100, -50), (100, 50), (0, 50), (0, -50)])]],
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        road = road.clip(private_property.iloc[0].geometry)
        private_property = private_property.overlay(road, how="difference").explode(ignore_index=True)
        buildings = gpd.GeoDataFrame(
            data=buildings,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        # OSM graph with a junction
        osm_graph = build_osm_test_graph(
            nodes=[(1, (0, 0)), (2, (50, 0)), (3, (100, 50)), (4, (100, 0))],
            edges=[(1, 2, 100), (2, 3, 101), (2, 4, 102)],
        )

        cost_surface_graph, cost_surface_nodes, pipe_ramming_crossings, pipe_ramming_settings = setup_theory_examples(
            road, buildings, private_property, trees, osm_graph, CrossingType.JUNCTION
        )

        self._run_crossing(
            cost_surface_graph,
            cost_surface_nodes,
            pipe_ramming_crossings,
            pipe_ramming_settings,
            expected_route_length_unprocessed,
            expected_route_length_straightened,
            n_expected_crossings,
            osm_graph,
            start_end,
            expected_route_cost,
            expected_crossings_used_in_route,
        )

    @pytest.mark.parametrize(
        [
            "buildings",
            "trees",
            "n_expected_crossings",
            "expected_route_length_unprocessed",
            "expected_route_length_straightened",
            "expected_route_cost",
            "expected_crossings_used_in_route",
            "start_end",
        ],
        [
            # Without obstacles
            (
                (),
                (),
                3,
                106,
                98,
                681.5,
                1,
                [(0.6, 6.5), (56, 49.5)],
            ),
            # With obstacles
            (
                [
                    [120, 0, shapely.LineString([(1, -18), (99, -18)]).buffer(8, cap_style="flat")],
                    [120, 0, shapely.LineString([(32, 10), (32, 46)]).buffer(8, cap_style="flat")],
                ],
                [[20, 0, shapely.Point(43.6, -8.2).buffer(6)], [20, shapely.Point(58, 19).buffer(4)]],
                2,
                123,
                109,
                838,
                2,
                [(0.6, 6.5), (99, -7)],
            ),
        ],
    )
    def test_theory_junction_degree_3_crossing_simple(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length_unprocessed,
        expected_route_length_straightened,
        expected_route_cost,
        expected_crossings_used_in_route,
        start_end,
        setup_theory_examples,
    ):
        road = (
            gpd.GeoDataFrame(
                data=[
                    # road east west
                    [30, 0, shapely.LineString([(0, 0), (100, 0)]).buffer(5, cap_style="flat")],
                    [5, 0, shapely.LineString([(0, 7), (100, 7)]).buffer(2, cap_style="flat")],
                    [5, 0, shapely.LineString([(0, -7), (100, -7)]).buffer(2, cap_style="flat")],
                    # road north
                    [5, 0, shapely.LineString([(43, 50), (43, 0)]).buffer(2, cap_style="flat")],
                    [30, 0, shapely.LineString([(50, 50), (50, 0)]).buffer(5, cap_style="flat")],
                    [5, 0, shapely.LineString([(57, 50), (57, 0)]).buffer(2, cap_style="flat")],
                ],
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
                crs=Config.CRS,
            )
            .dissolve(by="suitability_value")
            .explode()
            .reset_index()
        )
        private_property = gpd.GeoDataFrame(
            data=[[70, 0, shapely.Polygon([(0, -50), (100, -50), (100, 50), (0, 50), (0, -50)])]],
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        private_property = private_property.overlay(road, how="difference").explode(ignore_index=True)
        buildings = gpd.GeoDataFrame(
            data=buildings,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        # OSM graph with a junction
        osm_graph = build_osm_test_graph(
            nodes=[(1, (0, 0)), (2, (50, 0)), (3, (100, 0)), (4, (50, 50))],
            edges=[(1, 2, 100), (2, 3, 101), (2, 4, 102)],
        )

        cost_surface_graph, cost_surface_nodes, pipe_ramming_crossings, pipe_ramming_settings = setup_theory_examples(
            road, buildings, private_property, trees, osm_graph, CrossingType.JUNCTION
        )
        self._run_crossing(
            cost_surface_graph,
            cost_surface_nodes,
            pipe_ramming_crossings,
            pipe_ramming_settings,
            expected_route_length_unprocessed,
            expected_route_length_straightened,
            n_expected_crossings,
            osm_graph,
            start_end,
            expected_route_cost,
            expected_crossings_used_in_route,
        )

    @pytest.mark.parametrize(
        [
            "buildings",
            "trees",
            "n_expected_crossings",
            "expected_route_length_unprocessed",
            "expected_route_length_straightened",
            "expected_route_cost",
            "expected_crossings_used_in_route",
            "start_end",
        ],
        [
            # Without obstacles
            (
                (),
                (),
                4,
                118,
                109,
                803,
                2,
                [(0.3, -6.7), (56.23, 49.68)],
            ),
            # With obstacles
            (
                [
                    # south west
                    [120, 0, shapely.LineString([(1, -18), (40, -18)]).buffer(8, cap_style="flat")],
                    [120, 0, shapely.LineString([(32, -18), (32, -49)]).buffer(8, cap_style="flat")],
                    # north east
                    [120, 0, shapely.LineString([(1, 18), (40, 18)]).buffer(8, cap_style="flat")],
                    [120, 0, shapely.LineString([(32, 18), (32, 49)]).buffer(8, cap_style="flat")],
                    # north west
                    [120, 0, shapely.LineString([(68, 49), (68, 18)]).buffer(8, cap_style="flat")],
                    [120, 0, shapely.LineString([(60, 18), (99, 18)]).buffer(8, cap_style="flat")],
                    # south east
                    [120, 0, shapely.LineString([(68, -49), (68, -18)]).buffer(8, cap_style="flat")],
                    [120, 0, shapely.LineString([(60, -18), (99, -18)]).buffer(8, cap_style="flat")],
                ],
                [
                    [20, 0, shapely.Point(42.573, 8.294).buffer(4)],
                ],
                4,
                118,
                111,
                806.5,
                2,
                [(0.3, -6.7), (56.23, 49.68)],
            ),
        ],
    )
    def test_theory_junction_degree_4_crossing_simple(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length_unprocessed,
        expected_route_length_straightened,
        expected_route_cost,
        expected_crossings_used_in_route,
        start_end,
        setup_theory_examples,
    ):
        road = (
            gpd.GeoDataFrame(
                data=[
                    # road east west
                    [30, 0, shapely.LineString([(0, 0), (100, 0)]).buffer(5, cap_style="flat")],
                    [5, 0, shapely.LineString([(0, 7), (100, 7)]).buffer(2, cap_style="flat")],
                    [5, 0, shapely.LineString([(0, -7), (100, -7)]).buffer(2, cap_style="flat")],
                    # road north south
                    [5, 0, shapely.LineString([(50, -50), (50, 50)]).buffer(9, cap_style="flat")],
                    [30, 0, shapely.LineString([(50, -50), (50, 50)]).buffer(5, cap_style="flat")],
                ],
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
                crs=Config.CRS,
            )
            .dissolve(by=["suitability_value", "relatieveHoogteligging"])
            .explode()
            .reset_index()
        )
        private_property = gpd.GeoDataFrame(
            data=[[70, 0, shapely.Polygon([(0, -50), (100, -50), (100, 50), (0, 50), (0, -50)])]],
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        road = road.clip(private_property.iloc[0].geometry)
        private_property = private_property.overlay(road, how="difference").explode(ignore_index=True)
        buildings = (
            gpd.GeoDataFrame(
                data=buildings,
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
                crs=Config.CRS,
            )
            .dissolve(by=["suitability_value", "relatieveHoogteligging"])
            .explode()
            .reset_index()
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        # OSM graph with a junction
        osm_graph = build_osm_test_graph(
            nodes=[(1, (0, 0)), (2, (50, 0)), (3, (50, 50)), (4, (100, 0)), (5, (50, -50))],
            edges=[(1, 2, 100), (2, 3, 101), (2, 4, 102), (2, 5, 103)],
        )

        cost_surface_graph, cost_surface_nodes, pipe_ramming_crossings, pipe_ramming_settings = setup_theory_examples(
            road, buildings, private_property, trees, osm_graph, CrossingType.JUNCTION
        )

        self._run_crossing(
            cost_surface_graph,
            cost_surface_nodes,
            pipe_ramming_crossings,
            pipe_ramming_settings,
            expected_route_length_unprocessed,
            expected_route_length_straightened,
            n_expected_crossings,
            osm_graph,
            start_end,
            expected_route_cost,
            expected_crossings_used_in_route,
        )

    @pytest.mark.parametrize(
        [
            "buildings",
            "trees",
            "n_expected_crossings",
            "expected_route_length_unprocessed",
            "expected_route_length_straightened",
            "expected_route_cost",
            "expected_crossings_used_in_route",
            "start_end",
        ],
        [
            # Without obstacles
            (
                (),
                (),
                9,
                124,
                111,
                783.5,
                1,
                ((1, -3), (93, 45)),
            ),
            # With obstacles
            (
                [[120, 0, shapely.LineString([(2, -26), (44, -18), (63, -36)]).buffer(8, cap_style="flat")]],
                [[20, 0, shapely.Point(43, 6).buffer(4)]],
                9,
                138,
                123,
                913.5,
                2,
                [(1, -3), (93, 45)],
            ),
        ],
    )
    def test_theory_junction_degree_4_crossing_complex(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length_unprocessed,
        expected_route_length_straightened,
        expected_route_cost,
        expected_crossings_used_in_route,
        start_end,
        setup_theory_examples,
    ):
        road = (
            gpd.GeoDataFrame(
                data=[
                    # road north
                    [30, 0, shapely.LineString([(50, 50), (50, 0)]).buffer(5, cap_style="flat")],
                    [5, 0, shapely.LineString([(50, 50), (50, 0)]).buffer(7, cap_style="flat")],
                    # road north-east
                    [30, 0, shapely.LineString([(50, 0), (90, 50)]).buffer(5, cap_style="flat")],
                    [5, 0, shapely.LineString([(50, 0), (90, 50)]).buffer(7, cap_style="flat")],
                    # road south-west
                    [30, 0, shapely.LineString([(50, 0), (100, -50)]).buffer(5, cap_style="flat")],
                    [5, 0, shapely.LineString([(50, 0), (100, -50)]).buffer(7, cap_style="flat")],
                    # road west
                    [30, 0, shapely.LineString([(50, 0), (0, -10)]).buffer(5, cap_style="flat")],
                    [5, 0, shapely.LineString([(50, 0), (0, -10)]).buffer(7, cap_style="flat")],
                ],
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
                crs=Config.CRS,
            )
            .dissolve(by=["suitability_value", "relatieveHoogteligging"])
            .explode()
            .reset_index()
        )
        private_property = gpd.GeoDataFrame(
            data=[[70, 0, shapely.Polygon([(0, -50), (100, -50), (100, 50), (0, 50), (0, -50)])]],
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        road = road.clip(private_property.iloc[0].geometry)
        private_property = private_property.overlay(road, how="difference").explode(ignore_index=True)
        buildings = gpd.GeoDataFrame(
            data=buildings,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        # OSM graph with a junction
        osm_graph = build_osm_test_graph(
            nodes=[(1, (50, 0)), (2, (50, 50)), (3, (90, 50)), (4, (100, -50)), (5, (0, -10))],
            edges=[(1, 2, 105), (1, 3, 106), (1, 4, 107), (1, 5, 108)],
        )

        cost_surface_graph, cost_surface_nodes, pipe_ramming_crossings, pipe_ramming_settings = setup_theory_examples(
            road, buildings, private_property, trees, osm_graph, CrossingType.JUNCTION
        )

        self._run_crossing(
            cost_surface_graph,
            cost_surface_nodes,
            pipe_ramming_crossings,
            pipe_ramming_settings,
            expected_route_length_unprocessed,
            expected_route_length_straightened,
            n_expected_crossings,
            osm_graph,
            start_end,
            expected_route_cost,
            expected_crossings_used_in_route,
        )

    @pytest.mark.parametrize(
        [
            "buildings",
            "trees",
            "n_expected_crossings",
            "expected_route_length_unprocessed",
            "expected_route_length_straightened",
            "expected_route_cost",
            "expected_crossings_used_in_route",
            "start_end",
        ],
        [
            # Without obstacles
            (
                (),
                (),
                3,
                123,
                108,
                761.5,
                1,
                [(1, 8), (98, -6)],
            ),
            # With obstacles
            (
                [
                    [120, 0, shapely.LineString([(10, 15), (45, 15)]).buffer(5, cap_style="flat")],
                    [120, 0, shapely.LineString([(5, -15), (45, -15)]).buffer(5, cap_style="flat")],
                    [120, 0, shapely.LineString([(55, -15), (95, -15)]).buffer(5, cap_style="flat")],
                ],
                [[20, 0, shapely.Point(30, -8).buffer(4)], [20, shapely.Point(65, 9).buffer(4)]],
                3,
                123,
                108,
                761.5,
                1,
                [(1, 8), (98, -6)],
            ),
        ],
    )
    def test_theory_segment_crossing_straight_road(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length_unprocessed,
        expected_route_length_straightened,
        expected_route_cost,
        expected_crossings_used_in_route,
        start_end,
        setup_theory_examples,
    ):
        road = gpd.GeoDataFrame(
            data=[
                # Asphalt
                [30, 0, shapely.LineString([(0, 0), (100, 0)]).buffer(5, cap_style="flat")],
                # Pavement north
                [5, 0, shapely.LineString([(0, 7), (100, 7)]).buffer(2, cap_style="flat")],
                # Pavement south
                [5, 0, shapely.LineString([(0, -7), (100, -7)]).buffer(2, cap_style="flat")],
            ],
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        private_property = gpd.GeoDataFrame(
            data=[
                [70, 0, shapely.LineString([(0, 30), (100, 30)]).buffer(21, cap_style="flat")],
                [70, 0, shapely.LineString([(0, -30), (100, -30)]).buffer(21, cap_style="flat")],
            ],
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        buildings = gpd.GeoDataFrame(
            data=buildings,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        # MCDA vectors
        # Create a simple OSM graph, just one road.
        osm_graph = build_osm_test_graph(
            nodes=[(1, (0, 0)), (2, (100, 0))],
            edges=[(1, 2, 100)],
        )

        cost_surface_graph, cost_surface_nodes, pipe_ramming_crossings, pipe_ramming_settings = setup_theory_examples(
            road, buildings, private_property, trees, osm_graph, CrossingType.SEGMENT
        )

        self._run_crossing(
            cost_surface_graph,
            cost_surface_nodes,
            pipe_ramming_crossings,
            pipe_ramming_settings,
            expected_route_length_unprocessed,
            expected_route_length_straightened,
            n_expected_crossings,
            osm_graph,
            start_end,
            expected_route_cost,
            expected_crossings_used_in_route,
        )

    @pytest.mark.parametrize(
        [
            "buildings",
            "trees",
            "n_expected_crossings",
            "expected_route_length_unprocessed",
            "expected_route_length_straightened",
            "expected_route_cost",
            "expected_crossings_used_in_route",
            "start_end",
        ],
        [
            # Without obstacles
            (
                (),
                (),
                6,
                231,
                211,
                1385,
                1,
                [(1, 6), (159.4, -26.5)],
            ),
            # With obstacles
            (
                [
                    [
                        120,
                        0,
                        shapely.LineString([(55.4, 33.4), (80.5, -32)])
                        .buffer(11, cap_style="flat", single_sided=True)
                        .buffer(-1, cap_style="flat"),
                    ],
                    [
                        120,
                        0,
                        shapely.LineString([(69.4, -47.7), (48.3, 7.1)])
                        .buffer(11, cap_style="flat", single_sided=True)
                        .buffer(-1, cap_style="flat"),
                    ],
                    [
                        120,
                        0,
                        shapely.LineString([(17, 8), (53.5, 38.4)])
                        .buffer(11, cap_style="flat", single_sided=True)
                        .buffer(-1, cap_style="flat"),
                    ],
                ],
                [
                    [20, 0, shapely.Point(30, -8).buffer(4)],
                    [20, 0, shapely.Point(39.804, 6.896).buffer(5)],
                    [20, 0, shapely.Point(47.260, 10.421).buffer(6)],
                    [20, 0, shapely.Point(87, -30.9).buffer(4)],
                    [20, 0, shapely.Point(75.1, -47.6).buffer(6.5)],
                ],
                4,
                246,
                225,
                1481.5,
                1,
                [(1, 6), (159.4, -26.5)],
            ),
        ],
    )
    def test_theory_segment_crossing_complex_road(
        self,
        buildings,
        trees,
        n_expected_crossings,
        expected_route_length_unprocessed,
        expected_route_length_straightened,
        expected_route_cost,
        expected_crossings_used_in_route,
        start_end,
        setup_theory_examples,
    ):
        road_linestring = shapely.LineString([(0, 0), (20, 0), (50, 25), (75, -40), (120, -40), (150, -20), (160, -20)])
        road = (
            gpd.GeoDataFrame(
                data=[
                    [30, 0, road_linestring.buffer(5, cap_style="flat")],
                    [
                        5,
                        0,
                        shapely.difference(
                            road_linestring.buffer(8, cap_style="flat"),
                            road_linestring.buffer(5, cap_style="flat"),
                            grid_size=0.01,
                        ),
                    ],
                ],
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
                crs=Config.CRS,
            )
            .explode()
            .reset_index(drop=True)
        )
        private_property = gpd.GeoDataFrame(
            data=[
                [70, 0, shapely.LineString([(0, 0), (160, 0)]).buffer(90, cap_style="flat")],
            ],
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        private_property = private_property.overlay(road, how="difference").explode(ignore_index=True)
        buildings = gpd.GeoDataFrame(
            data=buildings,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        trees = gpd.GeoDataFrame(
            data=trees,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            crs=Config.CRS,
        )
        # MCDA vectors
        # Create a simple OSM graph, just one road.
        osm_graph = build_osm_test_graph(
            nodes=[
                (1, (0, 0)),
                (2, (20, 0)),
                (3, (50, 25)),
                (4, (75, -40)),
                (5, (120, -40)),
                (6, (150, -20)),
                (7, (160, -20)),
            ],
            edges=[
                (1, 2, 100),
                (2, 3, 101),
                (3, 4, 102),
                (4, 5, 103),
                (5, 6, 104),
                (6, 7, 105),
            ],
        )

        cost_surface_graph, cost_surface_nodes, pipe_ramming_crossings, pipe_ramming_settings = setup_theory_examples(
            road, buildings, private_property, trees, osm_graph, CrossingType.SEGMENT
        )

        self._run_crossing(
            cost_surface_graph,
            cost_surface_nodes,
            pipe_ramming_crossings,
            pipe_ramming_settings,
            expected_route_length_unprocessed,
            expected_route_length_straightened,
            n_expected_crossings,
            osm_graph,
            start_end,
            expected_route_cost,
            expected_crossings_used_in_route,
        )

    def _run_crossing(
        self,
        cost_surface_graph: rx.PyGraph,
        cost_surface_nodes: gpd.GeoDataFrame,
        pipe_ramming_crossings: list,
        pipe_ramming_settings: PipeRammingSettings,
        expected_route_length_unprocessed: float,
        expected_route_length_straightened: float,
        n_expected_crossings: int,
        osm_graph: rx.PyGraph,
        start_end: list[tuple],
        expected_route_cost: float = 0,
        expected_crossings_used_in_route: int = 0,
    ):
        multilayer_route_engine = MultilayerRouteEngine(
            cost_surface_graph,
            osm_graph,
            cost_surface_nodes,
            hexagon_size=self.hexagon_size,
            prefix=self.prefix,
            write_output=self.debug,
            out=self.out,
        )

        multilayer_route_engine.find_route(shapely.LineString(start_end))

        self._assert_crossings(
            pipe_ramming_crossings,
            cost_surface_nodes,
            pipe_ramming_settings.suitability_value_obstacles_threshold,
            multilayer_route_engine.hexagon_size,
            n_expected_crossings,
        )
        assert multilayer_route_engine.get_result_route_length_unprocessed() == pytest.approx(
            expected_route_length_unprocessed, abs=1
        )
        assert multilayer_route_engine.results.collapsed_linestring.length == pytest.approx(
            expected_route_length_straightened, abs=1
        )
        if expected_route_cost:
            assert multilayer_route_engine.get_result_route_cost() == expected_route_cost
        if expected_crossings_used_in_route:
            assert (
                len(
                    multilayer_route_engine.results.unprocessed_edges[
                        ~multilayer_route_engine.results.unprocessed_edges["origin"].isnull()
                    ]
                )
                == expected_crossings_used_in_route
            )

    @staticmethod
    def _assert_crossings(
        crossings: list,
        cost_surface_nodes: gpd.GeoDataFrame,
        threshold_suitability_value: float,
        hexagon_size: float,
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
                        cost_surface_nodes[
                            cost_surface_nodes.suitability_value > threshold_suitability_value
                        ].intersects(i[2].geometry.buffer(hexagon_size))
                    )
                ]
            )
            == 0
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
