# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pathlib
from datetime import datetime

import shapely
import geopandas as gpd
import time

import structlog
import typer
from structlog.contextvars import bound_contextvars

from settings import Config
from utility_route_planner.models.benchmark_routes import BenchmarkRouteCollection
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.multilayer_network.hexagon_graph.hexagon_graph_composer import build_and_compose_graph
from utility_route_planner.models.multilayer_network.multilayer_route_planner import MultilayerRouteEngine
from utility_route_planner.models.multilayer_network.osm_graph_downloader import OSMGraphDownloader
from utility_route_planner.models.multilayer_network.osm_graph_preprocessing import OSMGraphPreprocessor
from utility_route_planner.util.write import reset_geopackage

logger = structlog.get_logger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)


def run_multilayer_network(
    preset: str,
    path_geopackage_mcda_input: pathlib.Path,
    start_mid_end_points: shapely.LineString,
    project_area_geometry: shapely.Polygon,
    construction_date: datetime | None = None,
    prefix: str = "",
):
    start_cpu_time = time.process_time_ns()

    raw_graph = OSMGraphDownloader(project_area_geometry, graph_date=construction_date).download_graph()
    osm_graph_preprocessed = OSMGraphPreprocessor(raw_graph).preprocess_graph()

    mcda_engine = McdaCostSurfaceEngine(preset, path_geopackage_mcda_input, project_area_geometry)
    mcda_engine.preprocess_vectors()

    graph, gdf_nodes, pipe_ramming_crossings = build_and_compose_graph(
        processed_criteria_per_height_level=mcda_engine.processed_criteria_per_height_level,
        processed_criteria_vectors=mcda_engine.processed_vectors,
        raster_groups=mcda_engine.get_raster_groups(),
        project_area=mcda_engine.project_area_geometry,
        osm_graph_preprocessed=osm_graph_preprocessed,
    )

    multilayer_route_engine = MultilayerRouteEngine(
        graph,
        osm_graph_preprocessed,
        gdf_nodes,
        Config.HEXAGON_SIZE,
        prefix=prefix,
        write_output=True,
        experimental_smoothing=True,
    )
    multilayer_route_engine.find_route(start_mid_end_points)

    logger.info(f"Multilayer route CPU time: {(time.process_time_ns() - start_cpu_time) / 1e9:.2f} seconds.")


@app.command()
def run_debug_case():
    """e.g: uv run multilayer_main.py run-debug-case"""
    reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)

    with bound_contextvars(project_area="Componistenbuurt"):
        logger.info("Running multilayer routing engine for debug project area")
        run_multilayer_network(
            Config.RASTER_PRESET_NAME_BENCHMARK,
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            shapely.LineString([(174847.18, 451178.43), (175746.347, 450435.534)]),
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry,
        )


@app.command()
def run_benchmark_case(benchmark_case_ids: list[int] = typer.Argument(None)):
    """Example commands:

    Run all: uv run multilayer_main.py run-benchmark-case
    Run selected cases: uv run multilayer_main.py run-benchmark-case 2 3

    # TODO determine proper values for crossings
    # TODO add scores to the linestring (suitability score)
    """
    reset_geopackage(Config.PATH_GEOPACKAGE_MULTILAYER_NETWORK_OUTPUT, truncate=False)

    benchmark_routes = BenchmarkRouteCollection()
    if not benchmark_case_ids:
        benchmark_case_ids = benchmark_routes.get_all_case_ids()

    for benchmark_case_id in benchmark_case_ids:
        with bound_contextvars(benchmark_id=benchmark_case_id):
            reset_geopackage(Config.PATH_GEOPACKAGE_MCDA_OUTPUT, truncate=False)
            benchmark_route = benchmark_routes.get_case(benchmark_case_id)

            if benchmark_route.custom_project_area.is_empty:
                project_area = (
                    gpd.read_file(benchmark_route.path_geopackage, layer=benchmark_route.layer_name_project_area)
                    .iloc[0]
                    .geometry
                )
            else:
                project_area = benchmark_route.custom_project_area

            run_multilayer_network(
                Config.RASTER_PRESET_NAME_BENCHMARK,
                benchmark_route.path_geopackage,
                gpd.read_file(benchmark_route.path_geopackage, layer=benchmark_route.layer_name_human_designed_route)
                .iloc[0]
                .geometry,
                project_area,
                benchmark_route.construction_date,
                benchmark_route.raster_name_prefix,
            )


if __name__ == "__main__":
    app()
