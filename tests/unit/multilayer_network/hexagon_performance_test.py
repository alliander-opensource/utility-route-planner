# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import shapely
import geopandas as gpd

from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from settings import Config
from utility_route_planner.util.write import write_results_to_geopackage, reset_geopackage


class TestVectorToGraph:
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
    def ede_project_area(self) -> shapely.Polygon:
        return (
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry
        )

    @pytest.fixture()
    def project_area(self, larger_project_area: shapely.Polygon) -> shapely.Polygon:
        return larger_project_area

    @pytest.fixture()
    def vectors_for_project_areas(self, project_area: shapely.Polygon) -> McdaCostSurfaceEngine:
        mcda_engine = McdaCostSurfaceEngine(
            Config.RASTER_PRESET_NAME_BENCHMARK,
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            project_area,
        )
        mcda_engine.preprocess_vectors()
        return mcda_engine

    def test_vector_to_graph(
        self, vectors_for_project_areas: McdaCostSurfaceEngine, project_area: shapely.Polygon, debug: bool = True
    ):
        mcda_engine = vectors_for_project_areas

        raster_groups = {
            criteria_key: criteria.group for criteria_key, criteria in mcda_engine.raster_preset.criteria.items()
        }
        hexagon_graph_builder = HexagonGraphBuilder(
            mcda_engine.project_area_geometry,
            raster_groups,
            mcda_engine.processed_vectors,
            hexagon_size=0.5,
            block_size=64,
        )
        graph, nodes_gdf = hexagon_graph_builder.build_graph()

        if debug:
            edges_line_strings = []
            for source_node, target_node, _ in graph.edge_index_map().values():
                source_point = nodes_gdf.loc[nodes_gdf["node_id"] == source_node].geometry.iloc[0]
                target_point = nodes_gdf.loc[nodes_gdf["node_id"] == target_node].geometry.iloc[0]
                edges_line_strings.append(shapely.LineString([source_point, target_point]))
            edges = gpd.GeoDataFrame(geometry=edges_line_strings, crs=Config.CRS)
            reset_geopackage(Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT)
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, nodes_gdf, "graph_nodes", overwrite=True
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, edges, "graph_edges", overwrite=True
            )
            write_results_to_geopackage(
                Config.PATH_GEOPACKAGE_VECTOR_GRAPH_OUTPUT, project_area, "project_area", overwrite=True
            )
