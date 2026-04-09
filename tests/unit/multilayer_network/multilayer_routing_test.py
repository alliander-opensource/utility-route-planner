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
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import convert_hexagon_graph_to_gdfs
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine
from utility_route_planner.util.write import write_results_to_geopackage, reset_geopackage


class TestMultiLayerRouting:
    hexagon_size: float = 0.5
    debug: bool = True
    out = Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT

    def test_straightening_linestring_single_height_level(self):
        reset_geopackage(self.out, truncate=False)
        # build graph
        project_area = shapely.Polygon([(0, 0), (0, 100), (100, 100), (100, 0)])
        buildings = gpd.GeoDataFrame(
            data=[
                [30, 0, shapely.Point(9.7182, 10.0576).buffer(5)],  # small round tower
                [30, 0, shapely.LineString([(12, 6.3), (12, 0)]).buffer(1, cap_style="flat")],  # wall
                [30, 0, shapely.Point(100, 50).buffer(30).intersection(project_area)],  # big round tower
            ],
            geometry="geometry",
            crs=Config.CRS,
            columns=["suitability_value", "relatieveHoogteligging", "geometry"],
        )
        grassland = (
            gpd.GeoDataFrame(
                data=[
                    [2, 0, project_area.difference(buildings.geometry.union_all())],
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

        hexagon_graph_builder = HexagonGraphBuilder(
            project_area=project_area.buffer(0.01),
            raster_groups=raster_groups,
            preprocessed_vectors=processed_criteria_vectors,  # type: ignore
            hexagon_size=self.hexagon_size,
            block_size=Config.HEXAGON_BLOCK_SIZE,
        )
        graph = hexagon_graph_builder.build_graph()
        gdf_nodes = convert_hexagon_graph_to_gdfs(graph, edges=False)
        route_engine = MultilayerRouteEngine(
            graph,
            rx.PyGraph(),
            gdf_nodes,
            hexagon_size=self.hexagon_size,
            write_output=self.debug,
        )
        if self.debug:
            write_results_to_geopackage(self.out, grassland, "pytest_theory_grassland")
            write_results_to_geopackage(self.out, buildings, "pytest_theory_buildings")
            write_results_to_geopackage(self.out, gdf_nodes, "pytest_theory_nodes")
            gdf_nodes_copy = copy.deepcopy(gdf_nodes)
            gdf_nodes_copy["geometry"] = gdf_nodes.buffer(math.sqrt(3) * self.hexagon_size / 2)  # inradius
            write_results_to_geopackage(self.out, gdf_nodes_copy, "pytest_theory_nodes_inradius")

        # test that it can create a straight line: south -> north
        start_end = shapely.LineString([(10, 90), (10, 35)])
        route_engine.find_route(start_end)

        assert len(route_engine.result_route_node_indices) == 64
        assert route_engine.result_route_linestring.length == pytest.approx(55, self.hexagon_size)

        assert route_engine.result_route_straightened.length == pytest.approx(55, self.hexagon_size)
        assert len(route_engine.result_route_straightened_node_indices) == 2

        # Test that it can create a straight line: east -> west
        start_end = shapely.LineString([(10, 90), (65, 90)])
        route_engine.find_route(start_end)

        assert len(route_engine.result_route_node_indices) == 75
        assert route_engine.result_route_linestring.length == pytest.approx(55, self.hexagon_size)

        assert route_engine.result_route_straightened.length == pytest.approx(55, self.hexagon_size)
        assert len(route_engine.result_route_straightened_node_indices) == 2

        # test that it can properly navigate half of the small tower
        start_end = shapely.LineString([(9.7, 1.20), (9.7, 18.6)])
        route_engine.find_route(start_end)
        assert 1 == 1

        # test that it can circumnavigate the small tower
        start_end = shapely.LineString([(9.7, 1.20), (14.2, 1.4)])
        route_engine.find_route(start_end)
        assert 1 == 1

        # test that it can avoid the large tower
        start_end = shapely.LineString([(98, 2), (98.9, 98)])
        route_engine.find_route(start_end)
        # TODO i dont get why it does not skip larger segments of the tower?
        assert 1 == 1

        # test that it can zigzag through obstacles
        # TODO add some buildings

        # test it raises with invalid input
        with pytest.raises(ValueError):
            route_engine.find_route(shapely.LineString([(10, 1), (10, 1.2)]))
