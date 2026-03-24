# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pathlib

import shapely
import geopandas as gpd
import time

import structlog

from settings import Config
from utility_route_planner.models.benchmark_routes import BenchmarkRouteCollection
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_builder import HexagonGraphBuilder
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine
from utility_route_planner.models.multilayer_network.osm_graph_downloader import OSMGraphDownloader
from utility_route_planner.models.multilayer_network.osm_graph_preprocessing import OSMGraphPreprocessor
from utility_route_planner.models.multilayer_network.pipe_ramming import GetPotentialPipeRammingCrossings
from utility_route_planner.util.write import reset_geopackage

logger = structlog.get_logger(__name__)


def run_multilayer_network(
    preset: str,
    path_geopackage_mcda_input: pathlib.Path,
    start_mid_end_points: shapely.LineString,
    project_area_geometry: shapely.Polygon,
):
    reset_geopackage(Config.PATH_GEOPACKAGE_MCDA_OUTPUT, truncate=False)

    start_cpu_time = time.process_time_ns()

    raw_graph = OSMGraphDownloader(project_area_geometry).download_graph()
    osm_graph_preprocessed = OSMGraphPreprocessor(raw_graph).preprocess_graph()

    mcda_engine = McdaCostSurfaceEngine(preset, path_geopackage_mcda_input, project_area_geometry)
    mcda_engine.preprocess_vectors()

    raster_groups = {
        criteria_key: criteria.group for criteria_key, criteria in mcda_engine.raster_preset.criteria.items()
    }
    hexagon_graph_builder = HexagonGraphBuilder(
        mcda_engine.project_area_geometry,
        raster_groups,
        mcda_engine.processed_vectors,
        hexagon_size=Config.HEXAGON_SIZE,
        block_size=Config.HEXAGON_BLOCK_SIZE,
    )
    cost_surface_graph, cost_surface_nodes = hexagon_graph_builder.build_graph()
    pipe_ramming = GetPotentialPipeRammingCrossings(osm_graph_preprocessed, cost_surface_graph, cost_surface_nodes)
    _ = pipe_ramming.get_crossings()

    multi_layer_route_engine = MultilayerRouteEngine(
        pipe_ramming.cost_surface_graph, pipe_ramming.osm_graph, pipe_ramming.cost_surface_nodes, prefix="testing_"
    )
    multi_layer_route_engine.find_route(start_mid_end_points)

    logger.info(f"Multilayer route CPU time: {(time.process_time_ns() - start_cpu_time) / 1e9:.2f} seconds.")


if __name__ == "__main__":
    reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)

    # Small area
    # run_multilayer_network(
    #     Config.RASTER_PRESET_NAME_BENCHMARK,
    #     Config.PYTEST_PATH_GEOPACKAGE_MCDA,
    #     shapely.LineString([(174847.18,451178.43), (175746.347,450435.534)]),
    #     gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA).iloc[0].geometry
    # )

    benchmark_routes = BenchmarkRouteCollection()
    benchmark_route = benchmark_routes.route_2
    run_multilayer_network(
        Config.RASTER_PRESET_NAME_BENCHMARK,
        benchmark_route.path_geopackage,
        gpd.read_file(benchmark_route.path_geopackage, layer=benchmark_route.layer_name_human_designed_route)
        .iloc[0]
        .geometry,
        gpd.read_file(benchmark_route.path_geopackage, layer=benchmark_route.layer_name_project_area).iloc[0].geometry,
    )
