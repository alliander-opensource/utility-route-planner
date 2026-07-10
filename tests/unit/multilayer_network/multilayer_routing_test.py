# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import rustworkx as rx

from settings import Config
from tests.integration.conftest import write_criteria_vectors, write_cost_surface_nodes
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_composer import build_and_compose_graph
from utility_route_planner.models.multilayer_network.multilayer_route_helpers import get_inradius
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine, Algorithm
from utility_route_planner.util.write import reset_geopackage


class TestMultiLayerRouting:
    hexagon_size: float = 2.5
    debug: bool = Config.DEBUG
    out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT

    project_area = shapely.Polygon([(0, 0), (0, 100), (100, 100), (100, 0)]).buffer(0.01)

    @pytest.fixture
    def setup_grid(self):
        def _setup(building_collection: tuple = ()):
            buildings = gpd.GeoDataFrame(
                data=building_collection,
                geometry="geometry",
                crs=Config.CRS,
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            )
            grassland = (
                gpd.GeoDataFrame(
                    data=[
                        [2, 0, self.project_area.difference(buildings.geometry.union_all())],
                    ],
                    geometry="geometry",
                    crs=Config.CRS,
                    columns=["suitability_value", "relatieveHoogteligging", "geometry"],
                )
                .explode()
                .reset_index(drop=True)
            )
            raster_groups = {"grassland": "a", "buildings": "a"}
            processed_criteria_vectors = {"grassland": grassland, "buildings": buildings}
            processed_criteria_per_height_level = (
                {0: ["grassland", "buildings"]} if building_collection else {0: ["grassland"]}
            )

            graph, nodes_gdf, _ = build_and_compose_graph(
                processed_criteria_per_height_level=processed_criteria_per_height_level,
                processed_criteria_vectors=processed_criteria_vectors,
                raster_groups=raster_groups,
                project_area=self.project_area,
                debug=self.debug,
                hexagon_size=self.hexagon_size,
                block_size=512,
                apply_pipe_ramming=False,
            )

            if self.debug:
                reset_geopackage(self.out, truncate=False)
                write_criteria_vectors(self.project_area, processed_criteria_vectors, self.out)
                write_cost_surface_nodes(get_inradius(self.hexagon_size), nodes_gdf, self.out)

            route_engine = MultilayerRouteEngine(
                graph,
                rx.PyGraph(),
                nodes_gdf,
                hexagon_size=self.hexagon_size,
                write_output=self.debug,
                prefix="pytest_",
                algorithm=Algorithm.astar,
                experimental_smoothing=True,
            )
            return route_engine

        return _setup

    def test_straightening_linestring_no_obstacle(self, setup_grid):
        route_engine = setup_grid(())
        # test that it can create a straight line: south -> north
        start_end = shapely.LineString([(10, 90), (10, 10)])
        route_engine.find_route(start_end)
        assert len(route_engine.results.node_indices) == 20
        assert len(route_engine.results.collapsed_node_indices) == 2
        # Due to the flat top orientation, this is almost the same as the straightened line
        assert route_engine.results.unprocessed_linestring.length == pytest.approx(82.2, abs=0.1)
        assert route_engine.results.collapsed_linestring.length == pytest.approx(82.2, abs=0.1)
        assert (
            route_engine.results.quadratic_bezier_linestring.length == route_engine.results.collapsed_linestring.length
        )

        # Test that it can create a straight line: east -> west
        start_end = shapely.LineString([(10, 90), (93.468, 90.874)])
        route_engine.find_route(start_end)
        assert len(route_engine.results.node_indices) == 23
        assert len(route_engine.results.collapsed_node_indices) == 2
        assert route_engine.results.unprocessed_linestring.length == pytest.approx(95.3, abs=0.1)
        assert route_engine.results.collapsed_linestring.length == pytest.approx(82.5, abs=0.1)
        assert (
            route_engine.results.quadratic_bezier_linestring.length == route_engine.results.collapsed_linestring.length
        )

        # test that it can create a straight line: diagonal
        start_end = shapely.LineString([(10, 90), (90, 10)])
        route_engine.find_route(start_end)
        assert len(route_engine.results.node_indices) == 30
        assert len(route_engine.results.collapsed_node_indices) == 2
        assert route_engine.results.unprocessed_linestring.length == pytest.approx(125.5, abs=0.1)
        assert route_engine.results.collapsed_linestring.length == pytest.approx(112.3, abs=0.1)
        assert (
            route_engine.results.quadratic_bezier_linestring.length == route_engine.results.collapsed_linestring.length
        )

    def test_straightening_linestring_small_obstacle(self, setup_grid):
        route_engine = setup_grid(
            (
                [30, 0, shapely.Point(35, 50).buffer(15.1)],  # small round tower
                [30, 0, shapely.LineString([(12, 6.3), (12, 0)]).buffer(1, cap_style="flat")],  # wall
            )
        )
        # test that it can properly navigate half of the small tower
        start_end = shapely.LineString([(10, 50), (93.7, 53.7)])
        route_engine.find_route(start_end)
        assert len(route_engine.results.node_indices) == 24
        assert len(route_engine.results.collapsed_node_indices) == 3
        assert route_engine.results.unprocessed_linestring.length == pytest.approx(99.6, abs=0.1)
        assert route_engine.results.collapsed_linestring.length == pytest.approx(90.8, abs=0.1)
        assert route_engine.results.quadratic_bezier_linestring.length == pytest.approx(89.8, abs=0.1)

    def test_straightening_linestring_small_obstacle_circumnavigation(self, setup_grid):
        route_engine = setup_grid(
            (
                [30, 0, shapely.Point(33.665, 52.258).buffer(16.5)],  # small round tower
                [30, 0, shapely.LineString([(33.665, 52.258), (33.665, 100)]).buffer(5, cap_style="flat")],  # wall
            )
        )

        start_end = shapely.LineString([(10, 50), (44.710, 98.006)])
        route_engine.find_route(start_end)

        assert len(route_engine.results.node_indices) == 28
        assert len(route_engine.results.collapsed_node_indices) == 7
        assert route_engine.results.unprocessed_linestring.length == pytest.approx(116.9, abs=0.1)
        assert route_engine.results.collapsed_linestring.length == pytest.approx(107.3, abs=0.1)
        assert route_engine.results.quadratic_bezier_linestring.length == pytest.approx(106.1, abs=0.1)

    def test_straightening_linestring_large_obstacle_and_zigzags(self, setup_grid):
        route_engine = setup_grid(
            (
                [300, 0, shapely.Point(48.7, 52.2).buffer(35).intersection(self.project_area)],  # big round tower
                [300, 0, shapely.LineString([(0.056, 2.841), (27.773, 30.448)]).buffer(2, cap_style="flat")],
                [
                    300,
                    0,
                    shapely.LineString([(97.795, 6.253), (71.628, 17.343), (66.653, 4.718)]).buffer(
                        2, cap_style="flat"
                    ),
                ],
            )
        )
        # test that it can avoid the large tower and smaller walls
        start_end = shapely.LineString([(7.323, 51.3), (74.935, 10.679)])
        route_engine.find_route(start_end)

        assert len(route_engine.results.node_indices) == 50
        assert len(route_engine.results.collapsed_node_indices) == 12
        assert route_engine.results.unprocessed_linestring.length == pytest.approx(212.2, abs=0.1)
        assert route_engine.results.collapsed_linestring.length == pytest.approx(194, abs=0.1)
        assert route_engine.results.quadratic_bezier_linestring.length == pytest.approx(187.4, abs=0.1)

    def test_many_zigzags(self, setup_grid):
        inradius = get_inradius(self.hexagon_size)
        linestring = shapely.LineString(
            [(1.88, 76.72), (37.58, 97.7), (93.682, 64.830), (52.445, 41.017), (99.379, 14.247)]
        )
        route_engine = setup_grid(
            (
                [300, 0, shapely.Point(7.535, 62.606).buffer(10).intersection(self.project_area)],
                [300, 0, linestring.buffer(inradius, cap_style="flat")],
                [80, 0, shapely.Point(37.149, 79.696).buffer(self.hexagon_size)],
                [90, 0, shapely.Point(71.205, 64.922).buffer(self.hexagon_size)],
                [111, 0, shapely.Point(63.637, 61.002).buffer(self.hexagon_size)],
                [300, 0, linestring.offset_curve(-self.hexagon_size * 3).buffer(inradius, cap_style="flat")],
            )
        )

        start_end = shapely.LineString([(3.427, 73.255), (97.600, 10.573)])
        route_engine.find_route(start_end)

        assert len(route_engine.results.node_indices) == 46
        assert len(route_engine.results.collapsed_node_indices) == 7
        assert route_engine.results.unprocessed_linestring.length == pytest.approx(194.8, abs=0.1)
        assert route_engine.results.collapsed_linestring.length == pytest.approx(192.5, abs=0.1)
        assert route_engine.results.quadratic_bezier_linestring.length == pytest.approx(188.5, abs=0.1)
        assert route_engine.results.string_pulled_linestring.length == pytest.approx(184.8, abs=0.1)

    def test_minimum_bending_radius(self, setup_grid):
        inradius = get_inradius(self.hexagon_size)
        route_engine = setup_grid(
            (
                [300, 0, shapely.LineString([(39.316, 63.809), (66.870, 80.155)]).buffer(inradius, cap_style="flat")],
                [300, 0, shapely.Point(37.393, 62.681).buffer(inradius)],
                [300, 0, shapely.Point(37.326, 58.338).buffer(inradius)],
                [300, 0, shapely.Point(41.338, 56.182).buffer(inradius)],
                [300, 0, shapely.Point(44.853, 58.271).buffer(inradius)],
            )
        )
        route_engine.find_route(shapely.LineString([(41.073, 60.427), (48.699, 73.604)]))
        assert len(route_engine.results.node_indices) == 14
        assert len(route_engine.results.collapsed_node_indices) == 7

        assert route_engine.results.unprocessed_linestring.length == pytest.approx(56.29, abs=0.1)
        assert route_engine.results.collapsed_linestring.length == pytest.approx(50.49, abs=0.1)
        assert route_engine.results.quadratic_bezier_linestring.length == pytest.approx(46, abs=0.1)

    def test_multiple_segments_with_different_scores(self, setup_grid):
        route_engine = setup_grid(
            (
                [
                    10,
                    0,
                    shapely.LineString([(0, 80), (100, 80)])
                    .buffer(3, cap_style="flat")
                    .intersection(self.project_area),
                ],
                [
                    20,
                    0,
                    shapely.LineString([(0, 58), (100, 58)])
                    .buffer(3, cap_style="flat")
                    .intersection(self.project_area),
                ],
                [
                    30,
                    0,
                    shapely.LineString([(0, 40), (100, 40)])
                    .buffer(3, cap_style="flat")
                    .intersection(self.project_area),
                ],
                [
                    40,
                    0,
                    shapely.LineString([(0, 20), (100, 20)])
                    .buffer(3, cap_style="flat")
                    .intersection(self.project_area),
                ],
            )
        )
        # Straight crossing
        route_engine.find_route(shapely.LineString([(52, 1), (52, 99)]))
        assert len(route_engine.results.node_indices) == 23
        assert len(route_engine.results.collapsed_node_indices) == 10

        assert route_engine.results.unprocessed_linestring.length == pytest.approx(95.2, abs=0.1)
        assert route_engine.results.collapsed_linestring.length == pytest.approx(95.2, abs=0.1)
        assert route_engine.results.quadratic_bezier_linestring.length == pytest.approx(95.2, abs=0.1)

        # Diagonal crossing
        route_engine.find_route(shapely.LineString([(93.581, 95.256), (7.495, 6.328)]))
        assert len(route_engine.results.node_indices) == 33
        assert len(route_engine.results.collapsed_node_indices) == 10

        assert route_engine.results.unprocessed_linestring.length == pytest.approx(138.5, abs=0.1)
        assert route_engine.results.collapsed_linestring.length == pytest.approx(134.1, abs=0.1)
        assert route_engine.results.quadratic_bezier_linestring.length == pytest.approx(129.8, abs=0.1)

    @pytest.mark.skip(reason="Bezier curves do not respect suitability value.")
    def test_bezier_curvature(self, setup_grid):
        route_engine = setup_grid(([10, 0, shapely.Polygon([(0, 10), (90, 10), (90, 100), (0, 100)])],))
        # Straight crossing
        route_engine.find_route(shapely.LineString([(0, 1), (99, 99)]))
        # assert len(route_engine.results.node_indices) == 23
        assert len(route_engine.results.collapsed_node_indices) == 3
        # Note that Beziers currently do not respect suitability value integrity.
        assert not (
            route_engine.gdf_cost_surface_nodes[route_engine.gdf_cost_surface_nodes.suitability_value != 2]
            .buffer(get_inradius(route_engine.hexagon_size))
            .intersects(route_engine.results.quadratic_bezier_linestring)
            .any()
        )

    def test_invalid_input_route_engine(self, setup_grid):
        route_engine = setup_grid(())
        start_end = shapely.LineString([(10, 1), (10, 1.2)])
        with pytest.raises(ValueError):
            route_engine.find_route(start_end)
            route_engine.get_source_and_target_nodes(start_end)

    def setup_segments(self, edges):
        nodes = gpd.GeoDataFrame(
            data=[(i, shapely.Point()) for i in range(len(edges) + 1)],
            columns=["id", "geometry"],
        )
        route_engine = MultilayerRouteEngine(rx.PyGraph(), rx.PyGraph(), nodes, hexagon_size=self.hexagon_size)
        route_engine.results.unprocessed_edges = edges
        route_engine.results.node_indices = [i for i in range(len(edges) + 1)]
        return route_engine.get_segments()

    def test_create_segments_suitability_value_change(self):
        edges = pd.DataFrame(
            data=[
                (10, False, np.nan),
                (10, False, np.nan),
                (100, False, np.nan),
                (100, False, np.nan),
                (10, False, np.nan),
                (11, False, np.nan),
                (12, False, np.nan),
                (10, False, np.nan),
            ],
            columns=["weight", "connects_height_levels", "origin"],
        )
        segments = self.setup_segments(edges)

        assert segments["segment"].to_list() == [1, 1, 1, 2, 2, 3, 4, 5, 6]

    def test_create_segments_height_level_changes(self):
        edges = pd.DataFrame(
            data=[
                (10, False, np.nan),
                (10, False, np.nan),
                (10, False, np.nan),
                (10, True, np.nan),
                (10, False, np.nan),
                (10, False, np.nan),
                (10, True, np.nan),
                (10, False, np.nan),
            ],
            columns=["weight", "connects_height_levels", "origin"],
        )
        segments = self.setup_segments(edges)

        assert segments["segment"].to_list() == [1, 1, 1, 1, 2, 3, 3, 4, 5]

    def test_create_segments_mixed_changes(self):
        edges = pd.DataFrame(
            data=[
                (10, False, np.nan),
                (1222, False, "JUNCTION"),
                (10, False, np.nan),
                (10, False, np.nan),
                (130, False, np.nan),
                (130, False, np.nan),
                (10, False, np.nan),
                (10, False, np.nan),
                (99, False, np.nan),
                (10, False, np.nan),
                (263, False, "JUNCTION"),
                (301, False, "JUNCTION"),
                (10, False, np.nan),
                (10, False, np.nan),
            ],
            columns=["weight", "connects_height_levels", "origin"],
        )
        segments = self.setup_segments(edges)

        assert segments["segment"].to_list() == [1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 10]
