# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import shapely
import geopandas as gpd

from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from settings import Config
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_utils import convert_hexagon_graph_to_gdfs
from utility_route_planner.util.write import write_results_to_geopackage


class TestVectorToGraph:
    @pytest.fixture()
    def simple_project_area(self) -> shapely.Polygon:
        return shapely.Polygon(
            [
                shapely.Point(174992.960, 451097.964),
                shapely.Point(174993.753, 451088.943),
                shapely.Point(175004.559, 451089.438),
                shapely.Point(175005.154, 451097.468),
                shapely.Point(174992.960, 451097.964),
            ]
        )

    @pytest.fixture()
    def larger_project_area(self) -> shapely.Polygon:
        return shapely.Polygon(
            [
                shapely.Point(174932.067, 451134.757),
                shapely.Point(174921.054, 451035.046),
                shapely.Point(175021.659, 451031.772),
                shapely.Point(175026.123, 451131.483),
                shapely.Point(174932.067, 451134.757),
            ]
        )

    @pytest.fixture()
    def ede_project_area(self):
        return (
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry
        )

    @pytest.fixture()
    def vectors_for_project_areas(self, ede_project_area: shapely.Polygon) -> McdaCostSurfaceEngine:
        mcda_engine = McdaCostSurfaceEngine(
            Config.RASTER_PRESET_NAME_BENCHMARK,
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            ede_project_area,
        )
        mcda_engine.preprocess_vectors()
        return mcda_engine

    def test_vector_to_graph(self, vectors_for_project_areas: McdaCostSurfaceEngine, debug: bool = False):
        mcda_engine = vectors_for_project_areas

        raster_groups = {
            criteria_key: criteria.group for criteria_key, criteria in mcda_engine.raster_preset.criteria.items()
        }
        hexagon_graph_builder = HexagonGraphBuilder(
            mcda_engine.project_area_geometry,
            raster_groups,
            mcda_engine.processed_vectors,
            hexagon_size=0.5,
            block_size=128,
        )
        graph = hexagon_graph_builder.build_graph()

        if debug:
            nodes_gdf, edges_gdf = convert_hexagon_graph_to_gdfs(graph)
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, nodes_gdf, "graph_nodes", overwrite=True
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, edges_gdf, "graph_edges", overwrite=True
            )
