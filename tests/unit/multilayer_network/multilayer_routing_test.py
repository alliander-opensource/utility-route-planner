# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import copy

import geopandas as gpd
import pytest
import shapely
import math
import rustworkx as rx

from settings import Config
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_edge_generator import HexagonEdgeGenerator
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_grid_builder import HexagonGridBuilder
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine
from utility_route_planner.util.write import write_results_to_geopackage, reset_geopackage


class TestMultiLayerRouting:
    hexagon_size: float = 2.5
    debug: bool = True
    out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT

    project_area = shapely.Polygon([(0, 0), (0, 100), (100, 100), (100, 0)]).buffer(0.01)

    @pytest.fixture
    def setup_grid(self):
        def _setup(buildings: tuple = ()):
            _buildings = gpd.GeoDataFrame(
                data=buildings,
                geometry="geometry",
                crs=Config.CRS,
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            )
            grassland = (
                gpd.GeoDataFrame(
                    data=[
                        [2, 0, self.project_area.difference(_buildings.geometry.union_all())],
                    ],
                    geometry="geometry",
                    crs=Config.CRS,
                    columns=["suitability_value", "relatieveHoogteligging", "geometry"],
                )
                .explode()
                .reset_index(drop=True)
            )
            raster_groups = {"grassland": "a", "buildings": "a"}
            processed_criteria_vectors = {"grassland": grassland, "buildings": _buildings}

            grid_constructor = HexagonGridBuilder(hexagon_size=self.hexagon_size, block_size=Config.HEXAGON_BLOCK_SIZE)
            hexagon_edge_generator = HexagonEdgeGenerator()
            hexagon_graph_builder = HexagonGraphBuilder(
                grid_builder=grid_constructor, edge_generator=hexagon_edge_generator
            )
            graph, nodes_gdf = hexagon_graph_builder.build_graph(
                self.project_area,
                raster_groups,
                processed_criteria_vectors,  # type: ignore
            )

            if self.debug:
                reset_geopackage(self.out, truncate=False)
                write_results_to_geopackage(self.out, grassland, "pytest_theory_grassland")
                write_results_to_geopackage(self.out, _buildings, "pytest_theory_buildings")
                write_results_to_geopackage(self.out, nodes_gdf, "pytest_theory_nodes")

                gdf_nodes_copy = copy.deepcopy(nodes_gdf)
                gdf_nodes_copy["geometry"] = nodes_gdf.buffer(math.sqrt(3) * self.hexagon_size / 2)  # inradius
                write_results_to_geopackage(self.out, gdf_nodes_copy, "pytest_theory_nodes_inradius")

            route_engine = MultilayerRouteEngine(
                graph,
                rx.PyGraph(),
                nodes_gdf,
                hexagon_size=self.hexagon_size,
                write_output=self.debug,
            )
            return route_engine

        return _setup

    def test_straightening_linestring_no_obstacle(self, setup_grid):
        route_engine = setup_grid(())
        # test that it can create a straight line: south -> north
        start_end = shapely.LineString([(10, 90), (10, 10)])
        route_engine.find_route(start_end)
        assert len(route_engine.result_route_node_indices) == 20
        assert len(route_engine.result_route_straightened_node_indices) == 2
        # Due to the pointy top orientation, this is almost the same as the straightened line
        assert route_engine.result_route_linestring.length == pytest.approx(82.2, abs=0.1)
        assert route_engine.result_route_straightened.length == pytest.approx(82.2, abs=0.1)

        # Test that it can create a straight line: east -> west
        start_end = shapely.LineString([(10, 90), (90, 90)])
        route_engine.find_route(start_end)
        assert len(route_engine.result_route_node_indices) == 22
        assert route_engine.result_route_linestring.length == pytest.approx(90.9, abs=0.1)
        assert route_engine.result_route_straightened.length == pytest.approx(78.7, abs=0.1)
        assert len(route_engine.result_route_straightened_node_indices) == 2

        # test that it can create a straight line: diagonal
        start_end = shapely.LineString([(10, 90), (90, 10)])
        route_engine.find_route(start_end)
        assert len(route_engine.result_route_node_indices) == 30
        assert route_engine.result_route_linestring.length == pytest.approx(125.5, abs=0.1)
        assert route_engine.result_route_straightened.length == pytest.approx(112.3, abs=0.1)
        assert len(route_engine.result_route_straightened_node_indices) == 2

    def test_straightening_linestring_small_obstacle(self, setup_grid):
        route_engine = setup_grid(
            (
                [30, 0, shapely.Point(35, 50).buffer(15)],  # small round tower
                # [30, 0, shapely.LineString([(12, 6.3), (12, 0)]).buffer(1, cap_style="flat")],  # wall
            )
        )
        # test that it can properly navigate half of the small tower
        start_end = shapely.LineString([(10, 50), (93.7, 53.7)])
        route_engine.find_route(start_end)
        assert len(route_engine.result_route_node_indices) == 24
        assert route_engine.result_route_linestring.length == pytest.approx(99.6, abs=0.1)
        assert route_engine.result_route_straightened.length == pytest.approx(89.6, abs=0.1)
        assert len(route_engine.result_route_straightened_node_indices) == 3

    def test_straightening_linestring_small_obstacle_circumnavigation(self, setup_grid):
        route_engine = setup_grid(
            (
                [30, 0, shapely.Point(33.665, 52.258).buffer(16.5)],  # small round tower
                [30, 0, shapely.LineString([(33.665, 52.258), (33.665, 100)]).buffer(5, cap_style="flat")],  # wall
            )
        )

        start_end = shapely.LineString([(10, 50), (44.710, 98.006)])
        route_engine.find_route(start_end)

        assert len(route_engine.result_route_node_indices) == 28
        assert route_engine.result_route_linestring.length == pytest.approx(116.9, abs=0.1)
        assert route_engine.result_route_straightened.length == pytest.approx(107.3, abs=0.1)
        assert len(route_engine.result_route_straightened_node_indices) == 7

    def test_straightening_linestring_large_obstacle(self, setup_grid):
        route_engine = setup_grid(
            (
                [30, 0, shapely.Point(48.7, 52.2).buffer(35).intersection(self.project_area)],  # big round tower
                [10, 0, shapely.Point(91.022, 48.036).buffer(10)],  # smaller huddle
            )
        )
        # test that it can avoid the large tower
        start_end = shapely.LineString([(7.323, 51.3), (97.51, 51.3)])
        route_engine.find_route(start_end)

        assert route_engine.result_route_linestring.length == pytest.approx(142.9, abs=0.1)
        assert route_engine.result_route_straightened.length == pytest.approx(130.8, abs=0.1)
        assert len(route_engine.result_route_node_indices) == 32
        assert len(route_engine.result_route_straightened_node_indices) == 9

        # self.hexagon_size = 0.5
        # route_engine = setup_grid((
        #     [30, 0, shapely.Point(100, 50).buffer(35).intersection(self.project_area)],  # big round tower
        # )
        # )
        # route_engine.find_route(start_end)

    def test_straightening_linestring_obstacle_with_hole(self, setup_grid):
        # test that it can zigzag through obstacles
        # TODO add some buildings
        pass

    def test_straightening_linestring_obstacle_with_no_data(self, setup_grid):
        pass

    def test_invalid_input_route_engine(self, setup_grid):
        route_engine = setup_grid(())
        start_end = shapely.LineString([(10, 1), (10, 1.2)])
        with pytest.raises(ValueError):
            route_engine.find_route(start_end)
            route_engine.get_source_and_target_nodes(start_end)
