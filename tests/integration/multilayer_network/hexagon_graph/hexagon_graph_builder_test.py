# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import geopandas as gpd
import pytest
import shapely
import rustworkx as rx
from geopandas import GeoDataFrame
from shapely import Polygon

from settings import Config
from tests.integration.conftest import write_criteria_vectors
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_edge_generator import HexagonEdgeGenerator
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_composer import (
    build_and_compose_graph,
)
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_grid_builder import HexagonGridBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import convert_hexagon_edges_to_gdf
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine
from utility_route_planner.util.write import reset_geopackage, write_results_to_geopackage


HEXAGON_SIZE = 1
DEBUG = False
PREFIX = "pytest_"


class TestHexagonGraphBuilder:
    """
    This integration test tests whether artificially created vectors within a predefined project area are properly reflected
    in the hexagonal grid. First, the hexagonal grid for a single criterion is tested. Next, multiple criteria are used
    as input which enables more advanced testing with overlapping criteria.
    """

    @pytest.fixture()
    def hexagon_graph_builder(self) -> HexagonGraphBuilder:
        grid_constructor = HexagonGridBuilder(hexagon_size=HEXAGON_SIZE, block_size=Config.HEXAGON_BLOCK_SIZE)
        hexagon_edge_generator = HexagonEdgeGenerator()
        _hexagon_graph_builder = HexagonGraphBuilder(
            grid_builder=grid_constructor, edge_generator=hexagon_edge_generator
        )
        return _hexagon_graph_builder

    @pytest.fixture()
    def ede_project_area(self) -> shapely.Polygon:
        return shapely.get_geometry(
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry,
            0,
        )

    def test_build_graph_for_single_criterion(
        self,
        single_criterion_vectors: Callable,
        ede_project_area: shapely.Polygon,
        hexagon_graph_builder: HexagonGraphBuilder,
        debug: bool = DEBUG,
    ):
        max_value = Config.MAX_NODE_SUITABILITY_VALUE
        min_value = Config.MIN_NODE_SUITABILITY_VALUE
        single_criterion_vectors = single_criterion_vectors(max_value, min_value, max_value)

        # Create a simple vector dict for the single criterion.
        preprocessed_vectors = {"test": single_criterion_vectors}
        raster_criteria_groups = {"test": "a"}

        graph, nodes_gdf = hexagon_graph_builder.build_graph(
            ede_project_area, raster_criteria_groups, preprocessed_vectors
        )
        edges_gdf = convert_hexagon_edges_to_gdf(graph, nodes_gdf)

        sample_points = gpd.GeoDataFrame(
            data=[
                # Multiple overlapping values, take the max value
                [1, 10, shapely.Point(174871.877, 451084.402)],
                # Single vector, must be equal to vector suitability value
                [2, 5, shapely.Point(174868.877, 451086.134)],
                # Vector value exceeds max node value, must be reset to max value
                [3, max_value, shapely.Point(175012.877, 450908.599)],
                # Vector value lower than node min value, must be reset to min value
                [4, min_value, shapely.Point(175093.877, 450912.929)],
                # Vector value equal to max value, must remain the same
                [5, max_value, shapely.Point(174923.627, 450959.261)],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["sample_id", "expected_suitability_value", "geometry"],
        ).astype({"expected_suitability_value": "int16"})

        # Verify that the nodes near the sample points are equal to the expected value on the sample points.
        joined_sample_points = sample_points.sjoin_nearest(nodes_gdf)
        assert joined_sample_points["expected_suitability_value"].equals(joined_sample_points["suitability_value"])

        if debug:
            self.write_debug_output(
                ede_project_area, preprocessed_vectors, nodes_gdf, edges_gdf, sample_points, suffix="multiple_criterion"
            )

    def test_build_graph_for_multiple_criteria(
        self,
        multi_criteria_vectors: Callable,
        ede_project_area: shapely.MultiPolygon,
        hexagon_graph_builder: HexagonGraphBuilder,
        debug: bool = DEBUG,
    ):
        max_value = Config.MAX_NODE_SUITABILITY_VALUE
        min_value = Config.MIN_NODE_SUITABILITY_VALUE
        multiple_criteria_vectors = multi_criteria_vectors(max_value, min_value)

        raster_criteria_groups = {criterion_name: group for criterion_name, group, _ in multiple_criteria_vectors}
        preprocessed_vectors = {
            criterion_name: criterion_gdf for criterion_name, _, criterion_gdf in multiple_criteria_vectors
        }

        graph, nodes_gdf = hexagon_graph_builder.build_graph(
            ede_project_area, raster_criteria_groups, preprocessed_vectors
        )
        edges_gdf = convert_hexagon_edges_to_gdf(graph, nodes_gdf)

        sample_points = gpd.GeoDataFrame(
            data=[
                # Overlap between b1 and b2
                [1, 14.0, shapely.Point(175090.35, 450911.67)],
                # Overlap between a1, b1 and b2
                [2, min_value, shapely.Point(175091.8234, 450911.7488)],
                # Only b1
                [3, 1.0, shapely.Point(175088.2180, 450912.7950)],
                # Overlap between b1 and a1
                [4, max_value, shapely.Point(175013.3110, 450910.3013)],
                # Just a1
                [5, 5.0, shapely.Point(174839.089, 451050.785)],
                # Overlap between b1 and a1
                [6, 70.0, shapely.Point(174813.2646, 451113.9146)],
                # B1 and a1 sum is 0 here
                [7, 1.0, shapely.Point(174833.90, 451067.57)],
                # C1 overlaps a1
                [8, max_value, shapely.Point(174878.65, 451132.89)],
                # C1
                [9, max_value, shapely.Point(174799.54, 451170.54)],
                # C1 overlapping c1
                [10, max_value, shapely.Point(174921.44, 451123.59)],
                # C1 outside the project area
                [11, max_value, shapely.Point(174745.32, 451159.41)],
                # C2 overlapping b2
                [12, max_value, shapely.Point(175092.267, 450908.932)],
                # C2 overlapping b2, a1
                [13, max_value, shapely.Point(175097.673, 450912.390)],
                # C2 overlapping c1
                [14, max_value, shapely.Point(174847.32, 451177.96)],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["sample_id", "expected_suitability_value", "geometry"],
        ).astype({"expected_suitability_value": "int16"})

        # Verify that the nodes near the sample points are equal to the expected value on the sample points.
        joined_sample_points = sample_points.sjoin_nearest(nodes_gdf)
        assert joined_sample_points["expected_suitability_value"].equals(joined_sample_points["suitability_value"])

        if debug:
            self.write_debug_output(
                ede_project_area, preprocessed_vectors, nodes_gdf, edges_gdf, sample_points, suffix="multiple_criteria"
            )

    @staticmethod
    def write_debug_output(
        project_area: shapely.Polygon | shapely.MultiPolygon,
        criteria_vectors: dict[str, gpd.GeoDataFrame],
        nodes_gdf: gpd.GeoDataFrame,
        edges_gdf: gpd.GeoDataFrame,
        sample_points: gpd.GeoDataFrame,
        suffix: str,
    ):
        reset_geopackage(Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT)
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, project_area, "pytest_project_area", overwrite=True
        )
        write_criteria_vectors(project_area, criteria_vectors, Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT)
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT,
            nodes_gdf,
            f"pytest_graph_builder_integration_nodes_{suffix}",
            overwrite=True,
        )
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT,
            edges_gdf,
            f"pytest_graph_builder_integration_edges_{suffix}",
            overwrite=True,
        )
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT,
            sample_points,
            f"pytest_graph_builder_sample_points_{suffix}",
            overwrite=True,
        )


class TestHexagonGraphBuilderWithHeightLevels:
    out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT

    @pytest.fixture(autouse=True)
    def clean_start(self):
        if DEBUG:
            reset_geopackage(self.out, truncate=False)

    def test_build_graph_with_two_tunnels(self):
        """E.g., a road and a bicycle tunnel crossing each other."""
        project_area = shapely.Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])
        # Large road without sidewalks
        road_geom = shapely.LineString([(0, 50), (100, 50)])

        # Tunnel 1
        bicycle_road_north_geom_1 = shapely.LineString([(25, 80), (25, 100)])
        bicycle_tunnel_geom_1 = shapely.LineString([(25, 20), (25, 80)])
        bicycle_road_south_geom_1 = shapely.LineString([(25, 0), (25, 20)])

        # Tunnel 2
        bicycle_road_north_geom_2 = shapely.LineString([(75, 80), (75, 100)])
        bicycle_tunnel_geom_2 = shapely.LineString([(75, 20), (75, 80)])
        bicycle_road_south_geom_2 = shapely.LineString([(75, 0), (75, 20)])

        main_road = road_geom.buffer(10, cap_style="flat")
        road = gpd.GeoDataFrame(
            data=[
                [30, 0, main_road],
                [5, 1, bicycle_tunnel_geom_1.buffer(3, cap_style="flat")],
                [5, 0, bicycle_road_north_geom_1.buffer(3, cap_style="flat")],
                [5, 0, bicycle_road_south_geom_1.buffer(3, cap_style="flat")],
                [5, 1, bicycle_tunnel_geom_2.buffer(3, cap_style="flat")],
                [5, 0, bicycle_road_north_geom_2.buffer(3, cap_style="flat")],
                [5, 0, bicycle_road_south_geom_2.buffer(3, cap_style="flat")],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
        )
        grassland = (
            gpd.GeoDataFrame(
                data=[
                    [2, 0, project_area.difference(road[road["relatieveHoogteligging"] == 0].geometry.union_all())],
                ],
                geometry="geometry",
                crs=Config.CRS,
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            )
            .explode()
            .reset_index(drop=True)
        )
        processed_criteria_vectors = {
            "road": road,
            "grassland": grassland,
        }
        if DEBUG:
            write_criteria_vectors(project_area, processed_criteria_vectors)

        processed_criteria_per_height_level = {
            0: ["road", "grassland"],  # ground level
            1: ["road"],  # tunnel level
        }
        raster_groups = {
            "road": "a",
            "grassland": "a",
        }

        route_engine = self._initialize_multilayer_route_engine(
            processed_criteria_per_height_level, processed_criteria_vectors, project_area, raster_groups
        )

        # assert that we have a fully connected graph and no dangling parts.
        assert rx.number_connected_components(route_engine.cost_surface_graph) == 1

        # Find a route through the western tunnel
        route_engine.find_route(shapely.LineString([(6, 95), (6, 5)]))
        assert route_engine.get_result_route_length_unprocessed() == pytest.approx(109, 0.5)
        # assert we can route from north to south through a tunnel
        assert (
            len(route_engine.results.unprocessed_edges[route_engine.results.unprocessed_edges.connects_height_levels])
            == 2
        )
        # assert we did not cross the expensive road but used the tunnel
        assert all(route_engine.results.unprocessed_edges.weight < 30)
        # assert the number of connecting edges between height levels
        e = convert_hexagon_edges_to_gdf(route_engine.cost_surface_graph, route_engine.gdf_cost_surface_nodes)
        assert len(e[e.connects_height_levels]) == 48
        # assert we cannot skip halfway the tunnel to the main road.
        assert not all(e[e.connects_height_levels].intersects(main_road))
        # Because of route being on the western edge of the tunnel, it cannot shortcut as much as the eastern side.
        assert len(route_engine.results.collapsed_node_indices) == 6

        # Check that the other tunnel is working
        route_engine.find_route(shapely.LineString([(80, 95), (80, 5)]))
        assert route_engine.get_result_route_length_unprocessed() == pytest.approx(91, 0.5)
        assert not all(e[e.connects_height_levels].intersects(main_road))
        assert (
            len(route_engine.results.unprocessed_edges[route_engine.results.unprocessed_edges.connects_height_levels])
            == 2
        )
        assert len(route_engine.results.collapsed_node_indices) == 6

        # Find a route which does not use a tunnel but crosses the field above it.
        route_engine.find_route(shapely.LineString([(1, 65), (99, 65)]))
        assert route_engine.get_result_route_length_unprocessed() == pytest.approx(112, 0.5)
        assert (
            len(route_engine.results.unprocessed_edges[route_engine.results.unprocessed_edges.connects_height_levels])
            == 0
        )
        assert all(route_engine.results.unprocessed_edges.weight <= 4)
        assert len(route_engine.results.collapsed_node_indices) == 2

    def test_build_graph_with_one_bridge(self):
        """E.g., a road on a bridge crossing water. Could also be an ecoduct crossing a motorway."""
        project_area = shapely.Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])
        # Road on a bridge with sidewalks
        road_geom_west = shapely.LineString([(0, 50), (20, 50)])
        bridge_geom = shapely.LineString([(20, 50), (80, 50)])
        road_geom_east = shapely.LineString([(80, 50), (100, 50)])

        road = gpd.GeoDataFrame(
            data=[
                # west
                [10, 0, road_geom_west.buffer(10, cap_style="flat")],
                [5, 0, road_geom_west.offset_curve(12).buffer(2, cap_style="flat")],
                [5, 0, road_geom_west.offset_curve(-12).buffer(2, cap_style="flat")],
                # bridge
                [10, 1, bridge_geom.buffer(10, cap_style="flat")],
                [5, 1, bridge_geom.offset_curve(12).buffer(2, cap_style="flat")],
                [5, 1, bridge_geom.offset_curve(-12).buffer(2, cap_style="flat")],
                # east
                [10, 0, road_geom_east.buffer(10, cap_style="flat")],
                [5, 0, road_geom_east.offset_curve(12).buffer(2, cap_style="flat")],
                [5, 0, road_geom_east.offset_curve(-12).buffer(2, cap_style="flat")],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
        )
        # Water under the bridge
        water = gpd.GeoDataFrame(
            data=[
                [100, 0, shapely.LineString([(50, 0), (50, 100)]).buffer(10, cap_style="flat")],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
        )
        # Surrounding grassland
        gdf_street_height_0 = road[road["relatieveHoogteligging"] == 0]
        grassland = (
            gpd.GeoDataFrame(
                data=[
                    [
                        2,
                        0,
                        project_area.difference(gdf_street_height_0.geometry.union_all()).difference(
                            water.geometry.union_all()
                        ),
                    ],
                ],
                geometry="geometry",
                crs=Config.CRS,
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            )
            .explode()
            .reset_index(drop=True)
        )
        processed_criteria_vectors = {
            "road": road,
            "water": water,
            "grassland": grassland,
        }
        if DEBUG:
            # self.debug_write_output_vectors(project_area, processed_criteria_vectors)
            write_criteria_vectors(project_area, processed_criteria_vectors)

        # Expected output from mcda engine after changes
        processed_criteria_per_height_level = {
            0: ["road", "grassland", "water"],  # ground level
            1: ["road"],  # bridge level
        }
        raster_groups = {
            "road": "a",
            "water": "a",
            "grassland": "a",
        }

        route_engine = self._initialize_multilayer_route_engine(
            processed_criteria_per_height_level, processed_criteria_vectors, project_area, raster_groups
        )

        assert rx.number_connected_components(route_engine.cost_surface_graph) == 1
        e = convert_hexagon_edges_to_gdf(route_engine.cost_surface_graph, route_engine.gdf_cost_surface_nodes)
        assert len(e[e.connects_height_levels]) == 72

        # Find a route under the bridge
        route_engine.find_route(shapely.LineString([(6, 95), (6, 5)]))  # route should go under the bridge here (grass)
        assert route_engine.get_result_route_length_unprocessed() == pytest.approx(95, 0.5)
        # assert we can route from north to south through a tunnel
        assert (
            len(route_engine.results.unprocessed_edges[route_engine.results.unprocessed_edges.connects_height_levels])
            == 0
        )
        # assert we did not cross the expensive road or water but used the grass underneath the bridge.
        assert all(route_engine.results.unprocessed_edges.weight <= 4)

        # Find a route over the bridge
        route_engine.find_route(shapely.LineString([(1, 75), (99, 25)]))
        assert route_engine.get_result_route_length_unprocessed() == pytest.approx(147, 0.5)
        assert (
            len(route_engine.results.unprocessed_edges[route_engine.results.unprocessed_edges.connects_height_levels])
            == 2
        )
        # it should not cross water
        assert all(route_engine.results.unprocessed_edges.weight < 100)

    @staticmethod
    def _initialize_multilayer_route_engine(
        processed_criteria_per_height_level: dict[int, list[str]],
        processed_criteria_vectors: dict[str, GeoDataFrame],
        project_area: Polygon,
        raster_groups: dict[str, str],
    ) -> MultilayerRouteEngine:
        merged_graph, merged_nodes_gdf, _ = build_and_compose_graph(
            processed_criteria_per_height_level=processed_criteria_per_height_level,
            processed_criteria_vectors=processed_criteria_vectors,
            raster_groups=raster_groups,
            project_area=project_area,
            debug=DEBUG,
            hexagon_size=HEXAGON_SIZE,
            apply_pipe_ramming=False,
        )
        route_engine = MultilayerRouteEngine(
            merged_graph, rx.PyGraph(), merged_nodes_gdf, hexagon_size=HEXAGON_SIZE, write_output=DEBUG, prefix=PREFIX
        )
        return route_engine

    def test_build_graph_with_s_shaped_bridge_and_tunnel(self):
        """E.g., a road tunnel, a road and a bicycle bridge crossing each other."""
        project_area = shapely.Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])
        # Large road without sidewalks
        road_geom = shapely.LineString([(0, 50), (100, 50)])

        # Tunnel crossing the road
        tunnel_north = shapely.LineString([(50, 80), (50, 100)])
        tunnel_middle = shapely.LineString([(50, 20), (50, 80)])
        tunnel_south = shapely.LineString([(50, 0), (50, 20)])

        # Bridge crossing the tunnel
        bridge_north = shapely.LineString([(30, 80), (30, 100)])
        bridge_middle = shapely.LineString([(85, 20), (70, 20), (70, 50), (30, 50), (30, 80)])
        bridge_south = shapely.LineString([(85, 20), (100, 20)])

        road = gpd.GeoDataFrame(
            data=[
                [50, 0, road_geom.buffer(10, cap_style="flat")],
                [5, 0, tunnel_north.buffer(3, cap_style="flat")],
                [5, -1, tunnel_middle.buffer(3, cap_style="flat")],
                [5, 0, tunnel_south.buffer(3, cap_style="flat")],
                [5, 0, bridge_north.buffer(3, cap_style="flat")],
                [5, 1, bridge_middle.buffer(3, cap_style="flat", quad_segs=4)],
                [5, 0, bridge_south.buffer(3, cap_style="flat")],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
        ).clip(project_area)
        # Make grass a bit more expensive to force tunnel usage
        grassland = (
            gpd.GeoDataFrame(
                data=[
                    [6, 0, project_area.difference(road[road["relatieveHoogteligging"] == 0].geometry.union_all())],
                ],
                geometry="geometry",
                crs=Config.CRS,
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            )
            .explode()
            .reset_index(drop=True)
        )
        processed_criteria_vectors = {
            "road": road,
            "grassland": grassland,
        }
        if DEBUG:
            write_criteria_vectors(project_area, processed_criteria_vectors)
            # self.debug_write_output_vectors(project_area, processed_criteria_vectors)

        processed_criteria_per_height_level = {
            0: ["road", "grassland"],  # ground level
            1: ["road"],  # bridge level
            -1: ["road"],  # tunnel level
        }
        raster_groups = {
            "road": "a",
            "grassland": "a",
        }
        route_engine = self._initialize_multilayer_route_engine(
            processed_criteria_per_height_level, processed_criteria_vectors, project_area, raster_groups
        )

        assert rx.number_connected_components(route_engine.cost_surface_graph) == 1
        e = convert_hexagon_edges_to_gdf(route_engine.cost_surface_graph, route_engine.gdf_cost_surface_nodes)
        assert len(e[e.connects_height_levels]) == 51

        # find a route over the bridge, we do not cross grass
        route_engine.find_route(shapely.LineString([(30, 100), (100, 20)]))
        assert route_engine.get_result_route_length_unprocessed() == pytest.approx(145.3, 0.5)
        assert (
            len(route_engine.results.unprocessed_edges[route_engine.results.unprocessed_edges.connects_height_levels])
            == 2
        )
        assert all(route_engine.results.unprocessed_edges.weight <= 10)

        # find a route under the tunnel, we do not cross grass
        route_engine.find_route(shapely.LineString([(50, 100), (50, 0)]))
        assert route_engine.get_result_route_length_unprocessed() == pytest.approx(145.3, 0.5)
        assert (
            len(route_engine.results.unprocessed_edges[route_engine.results.unprocessed_edges.connects_height_levels])
            == 2
        )
        assert all(route_engine.results.unprocessed_edges.weight <= 10)

        # find a route with just grassland.
        route_engine.find_route(shapely.LineString([(1, 90), (99, 90)]))
        assert route_engine.get_result_route_length_unprocessed() == pytest.approx(112.4, 0.5)
        assert (
            len(route_engine.results.unprocessed_edges[route_engine.results.unprocessed_edges.connects_height_levels])
            == 0
        )
        assert all(route_engine.results.unprocessed_edges.weight <= 12)

    def test_build_graph_with_t_shaped_bridge(self):
        """E.g., a road and a bridge."""
        project_area = shapely.Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])
        # Road for cars
        road = shapely.LineString([(0, 50), (100, 50)])
        # Road on a bridge
        bridge_geom_south = shapely.LineString([(50, 20), (50, 50)])
        bridge_geom_north_west = shapely.LineString([(50, 50), (17, 80)])
        bridge_geom_north_east = shapely.LineString([(50, 50), (50, 80), (80, 80)])
        # Connecting parts to the bridge
        bridge_geom_south_0 = shapely.LineString([(50, 20), (50, 0)])
        bridge_geom_north_west_0 = shapely.LineString([(17, 80), (-5, 100)])
        bridge_geom_north_east_0 = shapely.LineString([(80, 80), (100, 80)])

        road = gpd.GeoDataFrame(
            data=[
                # road
                [50, 0, road.buffer(10, cap_style="flat")],
                # bridge
                [5, 1, bridge_geom_south.buffer(5, cap_style="flat")],
                [5, 1, bridge_geom_north_west.buffer(5, cap_style="flat")],
                [5, 1, bridge_geom_north_east.buffer(5, cap_style="flat")],
                # connecting parts to bridge
                [5, 0, bridge_geom_south_0.buffer(5, cap_style="flat")],
                [5, 0, bridge_geom_north_west_0.buffer(5, cap_style="flat")],
                [5, 0, bridge_geom_north_east_0.buffer(5, cap_style="flat")],
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
        ).clip(project_area)
        # Surrounding grassland
        gdf_street_height_0 = road[road["relatieveHoogteligging"] == 0]
        grassland = (
            gpd.GeoDataFrame(
                data=[
                    [6, 0, project_area.difference(gdf_street_height_0.geometry.union_all())],
                ],
                geometry="geometry",
                crs=Config.CRS,
                columns=["suitability_value", "relatieveHoogteligging", "geometry"],
            )
            .explode()
            .reset_index(drop=True)
        )
        processed_criteria_vectors = {
            "road": road,
            "grassland": grassland,
        }
        if DEBUG:
            write_criteria_vectors(project_area, processed_criteria_vectors)
            # self.debug_write_output_vectors(project_area, processed_criteria_vectors)

        # Expected output from mcda engine after changes
        processed_criteria_per_height_level = {
            0: ["road", "grassland"],  # ground level
            1: ["road"],  # bridge level
        }
        raster_groups = {
            "road": "a",
            "grassland": "a",
        }

        route_engine = self._initialize_multilayer_route_engine(
            processed_criteria_per_height_level, processed_criteria_vectors, project_area, raster_groups
        )

        assert rx.number_connected_components(route_engine.cost_surface_graph) == 1
        e = convert_hexagon_edges_to_gdf(route_engine.cost_surface_graph, route_engine.gdf_cost_surface_nodes)
        assert len(e[e.connects_height_levels]) == 47

        # Find a route over the bridge, both ways
        route_engine.find_route(shapely.LineString([(0, 95), (50, 0)]))
        assert route_engine.get_result_route_length_unprocessed() == pytest.approx(124.5, 0.5)
        assert (
            len(route_engine.results.unprocessed_edges[route_engine.results.unprocessed_edges.connects_height_levels])
            == 2
        )
        assert all(route_engine.results.unprocessed_edges.weight <= 10)

        route_engine.find_route(shapely.LineString([(100, 80), (50, 0)]))
        assert route_engine.get_result_route_length_unprocessed() == pytest.approx(128, 0.5)
        assert (
            len(route_engine.results.unprocessed_edges[route_engine.results.unprocessed_edges.connects_height_levels])
            == 2
        )
        assert all(route_engine.results.unprocessed_edges.weight <= 10)
