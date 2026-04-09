# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import geopandas as gpd
import pytest
import shapely
import rustworkx as rx

from settings import Config
from utility_route_planner.models.benchmark_routes import BenchmarkRouteCollection
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_edge_generator import HexagonEdgeGenerator
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_composer import (
    HexagonGraphComposer,
    HeightLevelGraph,
)
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_grid_builder import HexagonGridBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import convert_hexagon_edges_to_gdf
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine
from utility_route_planner.util.write import reset_geopackage, write_results_to_geopackage


@pytest.fixture()
def hexagon_graph_builder() -> HexagonGraphBuilder:
    hexagon_size = 1
    grid_constructor = HexagonGridBuilder(hexagon_size=hexagon_size, block_size=Config.HEXAGON_BLOCK_SIZE)
    hexagon_edge_generator = HexagonEdgeGenerator()
    _hexagon_graph_builder = HexagonGraphBuilder(grid_builder=grid_constructor, edge_generator=hexagon_edge_generator)
    return _hexagon_graph_builder


class TestHexagonGraphBuilder:
    """
    This integration test tests whether artificially created vectors within a predefined project area are properly reflected
    in the hexagonal grid. First, the hexagonal grid for a single criterion is tested. Next, multiple criteria are used
    as input which enables more advanced testing with overlapping criteria.
    """

    @pytest.fixture()
    def ede_project_area(self) -> shapely.MultiPolygon:
        return (
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry
        )

    def test_build_graph_for_single_criterion(
        self,
        single_criterion_vectors: Callable,
        ede_project_area: shapely.MultiPolygon,
        hexagon_graph_builder: HexagonGraphBuilder,
        debug: bool = False,
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
        debug: bool = False,
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
        project_area: shapely.MultiPolygon,
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
        for name, gdf in criteria_vectors.items():
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT,
                gdf,
                f"pytest_vector_{name}",
                overwrite=True,
            )

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
    debug: bool = False

    @pytest.fixture(autouse=True)
    def clean_start(self):
        if self.debug:
            reset_geopackage(self.out, truncate=False)

    def test_build_graph_with_two_tunnels(self, hexagon_graph_builder: HexagonGraphBuilder):
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
        if self.debug:
            self.debug_write_output_vectors(project_area, processed_criteria_vectors)

        processed_criteria_per_height_level = {
            0: ["road", "grassland"],  # ground level
            1: ["road"],  # tunnel level
        }
        raster_groups = {
            "road": "a",
            "grassland": "a",
        }

        merged_graph, merged_nodes_gdf = self._build_and_merge_graphs(
            self.debug,
            hexagon_graph_builder,
            processed_criteria_per_height_level,
            processed_criteria_vectors,
            project_area,
            raster_groups,
        )
        route_engine = MultilayerRouteEngine(merged_graph, rx.PyGraph(), merged_nodes_gdf, write_output=False)

        # assert that we have a fully connected graph and no dangling parts.
        assert rx.number_connected_components(merged_graph) == 1

        # Find a route through the western tunnel
        route_engine.find_route(shapely.LineString([(6, 95), (6, 5)]))
        assert route_engine.get_result_route_length() == pytest.approx(109, 0.5)
        # assert we can route from north to south through a tunnel
        assert len(route_engine.result_route_edges[route_engine.result_route_edges.connects_height_levels]) == 2
        # assert we did not cross the expensive road but used the tunnel
        assert all(route_engine.result_route_edges.weight < 30)
        # assert the number of connecting edges between height levels
        e = convert_hexagon_edges_to_gdf(merged_graph, merged_nodes_gdf)
        assert len(e[e.connects_height_levels]) == 48
        # assert we cannot skip halfway the tunnel to the main road.
        assert not all(e[e.connects_height_levels].intersects(main_road))

        # Check that the other tunnel is working
        route_engine.find_route(shapely.LineString([(80, 95), (80, 5)]))
        assert route_engine.get_result_route_length() == pytest.approx(91, 0.5)
        assert not all(e[e.connects_height_levels].intersects(main_road))
        assert len(route_engine.result_route_edges[route_engine.result_route_edges.connects_height_levels]) == 2

        # Find a route which does not use a tunnel but crosses the field above it.
        route_engine.find_route(shapely.LineString([(1, 65), (99, 65)]))
        assert route_engine.get_result_route_length() == pytest.approx(112, 0.5)
        # assert len(route_engine.result_route_edges[route_engine.result_route_edges.connects_height_levels]) == 0
        assert all(route_engine.result_route_edges.weight <= 4)

    def test_build_graph_with_one_bridge(self, hexagon_graph_builder: HexagonGraphBuilder):
        """E.g., a road on a bridge crossing water. Could also be an ecoduct crossing a motorway."""
        project_area = shapely.Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])
        # Road on a bridge with sidewalks
        road_geom_west = shapely.LineString([(0, 50), (25, 50)])
        bridge_geom = shapely.LineString([(25, 50), (75, 50)])
        road_geom_east = shapely.LineString([(75, 50), (100, 50)])

        road = gpd.GeoDataFrame(
            data=[
                # west
                [10, 0, road_geom_west.buffer(10, cap_style="flat")],
                [5, 0, road_geom_west.offset_curve(15).buffer(5, cap_style="flat")],
                [5, 0, road_geom_west.offset_curve(-15).buffer(5, cap_style="flat")],
                # bridge
                [10, 1, bridge_geom.buffer(10, cap_style="flat")],
                [5, 1, bridge_geom.offset_curve(15).buffer(5, cap_style="flat")],
                [5, 1, bridge_geom.offset_curve(-15).buffer(5, cap_style="flat")],
                # east
                [10, 0, road_geom_east.buffer(10, cap_style="flat")],
                [5, 0, road_geom_east.offset_curve(15).buffer(5, cap_style="flat")],
                [5, 0, road_geom_east.offset_curve(-15).buffer(5, cap_style="flat")],
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
        if self.debug:
            self.debug_write_output_vectors(project_area, processed_criteria_vectors)

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

        merged_graph, merged_nodes_gdf = self._build_and_merge_graphs(
            self.debug,
            hexagon_graph_builder,
            processed_criteria_per_height_level,
            processed_criteria_vectors,
            project_area,
            raster_groups,
        )

        route_engine = MultilayerRouteEngine(merged_graph, rx.PyGraph(), merged_nodes_gdf, write_output=self.debug)
        assert rx.number_connected_components(merged_graph) == 1
        e = convert_hexagon_edges_to_gdf(merged_graph, merged_nodes_gdf)
        assert len(e[e.connects_height_levels]) == 100

        # Find a route under the bridge
        route_engine.find_route(shapely.LineString([(6, 95), (6, 5)]))  # route should go under the bridge here (grass)
        assert route_engine.get_result_route_length() == pytest.approx(95, 0.5)
        # assert we can route from north to south through a tunnel
        # assert len(route_engine.result_route_edges[route_engine.result_route_edges.connects_height_levels]) == 0
        # assert we did not cross the expensive road or water but used the grass underneath the bridge.
        assert all(route_engine.result_route_edges.weight <= 4)

        # Find a route over the bridge
        route_engine.find_route(shapely.LineString([(1, 75), (99, 25)]))
        assert route_engine.get_result_route_length() == pytest.approx(147, 0.5)
        assert len(route_engine.result_route_edges[route_engine.result_route_edges.connects_height_levels]) == 2
        # it should not cross water
        assert all(route_engine.result_route_edges.weight < 100)

    def test_build_graph_with_s_shaped_bridge_and_tunnel(self, hexagon_graph_builder: HexagonGraphBuilder):
        """E.g., a road tunnel, a road and a bicycle bridge crossing each other."""
        project_area = shapely.Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])
        # Large road without sidewalks
        road_geom = shapely.LineString([(0, 50), (100, 50)])

        # Tunnel crossing the road
        tunnel_north = shapely.LineString([(50, 80), (50, 100)])
        tunnel_middle = shapely.LineString([(50, 20), (50, 80)])
        tunnel_south = shapely.LineString([(50, 0), (50, 20)])

        # Bridge crossing the tunnel
        bridge_north = shapely.LineString([(30, 70), (30, 100)])
        bridge_middle = shapely.LineString([(70, 30), (70, 50), (30, 50), (30, 70)])
        bridge_south = shapely.LineString([(70, 0), (70, 30)])

        road = gpd.GeoDataFrame(
            data=[
                [10, 0, road_geom.buffer(10, cap_style="flat")],
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
        if self.debug:
            self.debug_write_output_vectors(project_area, processed_criteria_vectors)

        processed_criteria_per_height_level = {
            0: ["road", "grassland"],  # ground level
            1: ["road"],  # bridge level
            -1: ["road"],  # tunnel level
        }
        raster_groups = {
            "road": "a",
            "grassland": "a",
        }
        merged_graph, merged_nodes_gdf = self._build_and_merge_graphs(
            self.debug,
            hexagon_graph_builder,
            processed_criteria_per_height_level,
            processed_criteria_vectors,
            project_area,
            raster_groups,
        )

        # route_engine = MultilayerRouteEngine(merged_graph, rx.PyGraph(), merged_nodes_gdf, write_output=self.debug)

        assert rx.number_connected_components(merged_graph) == 1
        # e = convert_hexagon_edges_to_gdf(merged_graph, merged_nodes_gdf)
        # assert len(e[e.connects_height_levels]) == 100

        # find a route over the bridge
        # route_engine.find_route()

        # find a route under the tunnel

        # find a route with just grassland

    def test_build_graph_with_t_shaped_bridge_height_levels(self):
        pass

    def test_example_data_integration(self, hexagon_graph_builder: HexagonGraphBuilder):
        """Use for testing a specific area of the example geopackages with known bridges/tunnels."""
        reset_geopackage(Config.PATH_GEOPACKAGE_MCDA_OUTPUT, truncate=False)
        project_area = shapely.Point(187224.708, 429010.295).buffer(200)
        mcda_engine = McdaCostSurfaceEngine(
            Config.RASTER_PRESET_NAME_BENCHMARK,
            BenchmarkRouteCollection.route_4.path_geopackage,
            project_area,
            raster_name_prefix="pytest_",
        )
        mcda_engine.preprocess_vectors()

        raster_groups = {
            criteria_key: criteria.group for criteria_key, criteria in mcda_engine.raster_preset.criteria.items()
        }
        merged_graph, merged_nodes_gdf = self._build_and_merge_graphs(
            self.debug,
            hexagon_graph_builder,
            mcda_engine.processed_criteria_per_height_level,
            mcda_engine.processed_vectors,
            project_area,
            raster_groups,
        )

        route_engine = MultilayerRouteEngine(merged_graph, rx.PyGraph(), merged_nodes_gdf, write_output=self.debug)
        route_engine.find_route(
            shapely.LineString([(187174.77, 429021.37), (187259.45, 429011.20)])
        )  # route should go under

    def debug_write_output_vectors(
        self,
        project_area: shapely.MultiPolygon | shapely.Polygon,
        criteria_vectors: dict[str, gpd.GeoDataFrame],
    ):
        write_results_to_geopackage(self.out, project_area, "pytest_project_area", overwrite=True)
        for name, gdf in criteria_vectors.items():
            write_results_to_geopackage(
                self.out,
                gdf,
                f"pytest_vector_{name}",
                overwrite=True,
            )

    def _build_and_merge_graphs(
        self,
        debug,
        hexagon_graph_builder,
        processed_criteria_per_height_level,
        processed_criteria_vectors,
        project_area,
        raster_groups,
    ) -> tuple[rx.PyGraph, gpd.GeoDataFrame]:
        # TODO extract to hex builder? Cache the project area node grid so it is not recomputed each time
        # Build hexagon graphs per height level
        graphs_per_height: dict[int, HeightLevelGraph] = {}
        for height_level, criteria in processed_criteria_per_height_level.items():
            criteria_for_height_level = {}
            for criterion in criteria:
                gdf = processed_criteria_vectors[criterion][
                    processed_criteria_vectors[criterion]["relatieveHoogteligging"] == height_level
                ]
                criteria_for_height_level[criterion] = gdf  # type: ignore

            graph, nodes_gdf = hexagon_graph_builder.build_graph(
                project_area.buffer(0.01), raster_groups, criteria_for_height_level
            )

            graphs_per_height[height_level] = HeightLevelGraph(graph, nodes_gdf)
        # if debug:
        #     # TODO fix
        #     self.debug_write_output_graphs(graphs_per_height)

        hexagon_graph_composer = HexagonGraphComposer(
            processed_criteria_per_height_level,
            graphs_per_height,
            hexagon_size=hexagon_graph_builder.hexagon_size,
            debug=debug,
        )
        merged_graph = hexagon_graph_composer.compose()

        return merged_graph.graph, merged_graph.nodes_gdf

    def debug_write_output_graphs(self, graphs: dict[int, HeightLevelGraph]):
        for height_level, height_level_graph in graphs.items():
            edges_gdf = convert_hexagon_edges_to_gdf(height_level_graph.graph, height_level_graph.nodes_gdf)
            write_results_to_geopackage(
                self.out,
                height_level_graph.nodes_gdf,
                f"pytest_graph_nodes_height_level_{height_level}",
                overwrite=True,
            )
            write_results_to_geopackage(
                self.out, edges_gdf, f"pytest_graph_edges_height_level_{height_level}", overwrite=True
            )
