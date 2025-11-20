# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import geopandas as gpd
import pytest
import shapely
import rustworkx as rx

from settings import Config
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_composer import HexagonGraphComposer
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import convert_hexagon_graph_to_gdfs
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine
from utility_route_planner.util.write import reset_geopackage, write_results_to_geopackage


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
        debug: bool = False,
    ):
        max_value = Config.MAX_NODE_SUITABILITY_VALUE
        min_value = Config.MIN_NODE_SUITABILITY_VALUE
        single_criterion_vectors = single_criterion_vectors(max_value, min_value, max_value)

        # Create a simple vector dict for the single criterion.
        preprocessed_vectors = {"test": single_criterion_vectors}
        raster_criteria_groups = {"test": "a"}

        hexagon_graph_builder = HexagonGraphBuilder(
            ede_project_area,
            raster_criteria_groups,
            preprocessed_vectors,
            hexagon_size=0.5,
        )
        graph = hexagon_graph_builder.build_graph()
        nodes_gdf, edges_gdf = convert_hexagon_graph_to_gdfs(graph)

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
        )

        # Verify that the nodes near the sample points are equal to the expected value on the sample points.
        joined_sample_points = sample_points.sjoin_nearest(nodes_gdf)
        assert joined_sample_points["expected_suitability_value"].equals(joined_sample_points["suitability_value"])

        if debug:
            self.write_debug_output(ede_project_area, preprocessed_vectors, nodes_gdf, edges_gdf, sample_points)

    def test_build_graph_for_multiple_criteria(
        self, multi_criteria_vectors: Callable, ede_project_area: shapely.MultiPolygon, debug: bool = False
    ):
        max_value = Config.MAX_NODE_SUITABILITY_VALUE
        min_value = Config.MIN_NODE_SUITABILITY_VALUE
        multiple_criteria_vectors = multi_criteria_vectors(max_value, min_value)

        raster_criteria_groups = {criterion_name: group for criterion_name, group, _ in multiple_criteria_vectors}
        preprocessed_vectors = {
            criterion_name: criterion_gdf for criterion_name, _, criterion_gdf in multiple_criteria_vectors
        }

        hexagon_graph_builder = HexagonGraphBuilder(
            ede_project_area,
            raster_criteria_groups,
            preprocessed_vectors,
            hexagon_size=0.5,
        )

        graph = hexagon_graph_builder.build_graph()
        nodes_gdf, edges_gdf = convert_hexagon_graph_to_gdfs(graph)

        sample_points = gpd.GeoDataFrame(
            data=[
                # Overlap between b1 and b2
                [1, 14.0, shapely.Point(175090.35, 450911.67)],
                # Overlap between a1, b1 and b2
                [2, min_value, shapely.Point(175091.8234, 450911.7488)],
                # Only b1
                [3, -1.0, shapely.Point(175088.2180, 450912.7950)],
                # Overlap between b1 and a1
                [4, max_value, shapely.Point(175013.3110, 450910.3013)],
                # Just a1
                [5, 5.0, shapely.Point(174839.089, 451050.785)],
                # Overlap between b1 and a1
                [6, 70.0, shapely.Point(174813.2646, 451113.9146)],
                # B1 and a1 sum is 0 here
                [7, 0.0, shapely.Point(174833.90, 451067.57)],
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
        )

        # Verify that the nodes near the sample points are equal to the expected value on the sample points.
        joined_sample_points = sample_points.sjoin_nearest(nodes_gdf)
        assert joined_sample_points["expected_suitability_value"].equals(joined_sample_points["suitability_value"])

        if debug:
            self.write_debug_output(ede_project_area, preprocessed_vectors, nodes_gdf, edges_gdf, sample_points)

    @staticmethod
    def write_debug_output(
        project_area: shapely.MultiPolygon,
        criteria_vectors: dict[str, gpd.GeoDataFrame],
        nodes_gdf: gpd.GeoDataFrame,
        edges_gdf: gpd.GeoDataFrame,
        sample_points: gpd.GeoDataFrame,
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
            Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, nodes_gdf, "pytest_graph_nodes", overwrite=True
        )
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, edges_gdf, "pytest_graph_edges", overwrite=True
        )
        write_results_to_geopackage(
            Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, sample_points, "pytest_points_to_sample", overwrite=True
        )


class TestHexagonGraphBuilderWithHeightLevels:
    out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT
    hexagon_size = 2.1

    @pytest.fixture(autouse=True)
    def clean_start(self):
        reset_geopackage(self.out, truncate=False)

    def test_build_graph_with_a_tunnels_with_osm(self):
        """E.g., a road and a bicycle tunnel crossing each other."""
        pass

    def test_build_graph_with_a_bridge_without_osm(self, debug: bool = True):
        """E.g., a road on a bridge crossing water. Could also be an ecoduct crossing a motorway."""
        project_area = shapely.Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])
        # Road on a bridge with pavement
        road_geom_west = shapely.LineString([(0, 50), (30, 50)])
        bridge_geom = shapely.LineString([(30, 50), (70, 50)])
        road_geom_east = shapely.LineString([(70, 50), (100, 50)])
        road_geom = shapely.line_merge(shapely.MultiLineString([road_geom_west, bridge_geom, road_geom_east]))
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
        if debug:
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

        # Build hexagon graphs per height level
        # TODO extract to hex builder? Cache the project area node grid so it is not recomputed each time
        graphs_per_height = {}
        for height_level, criteria in processed_criteria_per_height_level.items():
            processed_criteria_per_height_level = {}
            for criterion in criteria:
                gdf = processed_criteria_vectors[criterion][
                    processed_criteria_vectors[criterion]["relatieveHoogteligging"] == height_level
                ]
                processed_criteria_per_height_level[criterion] = gdf  # type: ignore

            hexagon_graph_builder = HexagonGraphBuilder(
                project_area=project_area,
                raster_groups=raster_groups,
                preprocessed_vectors=processed_criteria_per_height_level,  # type: ignore
                hexagon_size=self.hexagon_size,
            )
            graph = hexagon_graph_builder.build_graph()
            graphs_per_height[height_level] = graph
        if debug:
            self.debug_write_output_graphs(graphs_per_height)

        hexagon_graph_composer = HexagonGraphComposer(
            processed_criteria_per_height_level,
            graphs_per_height,
            hexagon_size=self.hexagon_size,
            gdf_osm_edges=gpd.GeoDataFrame(gpd.GeoSeries(road_geom), columns=["geometry"], crs=Config.CRS),
            debug=debug,
        )
        merged_graph = hexagon_graph_composer.compose()

        route_engine = MultilayerRouteEngine(
            merged_graph, rx.PyGraph(), hexagon_graph_composer.gdf_main_nodes, write_output=debug
        )
        route_engine.find_route(shapely.LineString([(6, 95), (6, 5)]))  # route should go under the bridge here (grass)
        route_engine.find_route(shapely.LineString([(3, 65), (98, 95)]))  # route should go over the bridge here

    def test_build_graph_with_multiple_height_levels_with_osm(self):
        """E.g., a road tunnel, a road and a bicycle bridge crossing each other."""
        pass

    def test_build_graph_with_multiple_height_levels_without_osm(self):
        """E.g., a road tunnel, a road and an ecoduct crossing each other."""
        pass

    def debug_write_output_vectors(
        self,
        project_area: shapely.MultiPolygon,
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

    def debug_write_output_graphs(
        self,
        graphs=dict[int, rx.PyGraph],
    ):
        for height_level, graph in graphs.items():
            nodes_gdf, edges_gdf = convert_hexagon_graph_to_gdfs(graph)
            write_results_to_geopackage(
                self.out, nodes_gdf, f"pytest_graph_nodes_height_level_{height_level}", overwrite=True
            )
            write_results_to_geopackage(
                self.out, edges_gdf, f"pytest_graph_edges_height_level_{height_level}", overwrite=True
            )
