# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pathlib
import time

import shapely
import structlog
import typer
from structlog.contextvars import bound_contextvars

from utility_route_planner.models.benchmark_routes import BenchmarkRouteCollection
from utility_route_planner.models.lcpa.lcpa_engine import LcpaUtilityRouteEngine
from settings import Config
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.route_evaluation_metrics import RouteEvaluationMetrics
from utility_route_planner.util.geo_utilities import get_first_last_point_from_linestring
from utility_route_planner.util.write import reset_geopackage
import geopandas as gpd

logger = structlog.get_logger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)


def run_mcda_lcpa(
    preset: str,
    path_geopackage_mcda_input: pathlib.Path,
    project_area_geometry: shapely.Polygon,
    start_mid_end_points: tuple[shapely.Point, ...],
    human_designed_route: shapely.LineString,
    raster_name_prefix: str,
    compute_rasters_in_parallel: bool,
    compute_metrics: bool = True,
):
    reset_geopackage(Config.PATH_GEOPACKAGE_MCDA_OUTPUT, truncate=False)

    start_cpu_time = time.process_time_ns()

    mcda_engine = McdaCostSurfaceEngine(preset, path_geopackage_mcda_input, project_area_geometry, raster_name_prefix)
    mcda_engine.preprocess_vectors()
    path_suitability_raster = mcda_engine.preprocess_rasters(
        mcda_engine.processed_vectors,
        cell_size=Config.RASTER_CELL_SIZE,
        max_block_size=Config.MAX_BLOCK_SIZE,
        run_in_parallel=compute_rasters_in_parallel,
    )

    lcpa_engine = LcpaUtilityRouteEngine()
    lcpa_engine.get_lcpa_route(
        path_suitability_raster,
        shapely.LineString(start_mid_end_points),
        mcda_engine.raster_preset.general.project_area_geometry,
    )

    logger.info(f"Route CPU time: {(time.process_time_ns() - start_cpu_time) / 1e9:.2f} seconds.")
    if compute_metrics:
        route_evaluation_metrics = RouteEvaluationMetrics(
            lcpa_engine.lcpa_result,
            path_suitability_raster,
            human_designed_route,
            project_area_geometry,
            mcda_engine.processed_vector_metrics,
        )
        route_evaluation_metrics.get_route_evaluation_metrics()


@app.command()
def run_benchmark_cases(benchmark_case_id: int | None = None):
    """
    Running:
        - Specific case: uv run main.py --benchmark-case-id 4
        - all cases: uv run main.py
    """
    reset_geopackage(Config.PATH_GEOPACKAGE_LCPA_OUTPUT, truncate=True)

    benchmark_routes = BenchmarkRouteCollection()
    if benchmark_case_id is None:
        benchmark_case_ids = benchmark_routes.get_all_case_ids()
    else:
        benchmark_case_ids = [benchmark_case_id]

    for benchmark_case_id in benchmark_case_ids:
        benchmark_route = benchmark_routes.get_case(benchmark_case_id)

        human_designed_route = (
            gpd.read_file(
                benchmark_route.path_geopackage,
                layer=benchmark_route.layer_name_human_designed_route,
            )
            .iloc[0]
            .geometry
        )
        route_stops = get_first_last_point_from_linestring(human_designed_route)
        if benchmark_route.stops:
            route_stops = tuple(
                list(route_stops)[:1] + [shapely.Point(i) for i in benchmark_route.stops] + list(route_stops)[1:]
            )

        with bound_contextvars(benchmark_id=benchmark_case_id):
            run_mcda_lcpa(
                preset=Config.RASTER_PRESET_NAME_BENCHMARK,
                path_geopackage_mcda_input=benchmark_route.path_geopackage,
                project_area_geometry=gpd.read_file(
                    benchmark_route.path_geopackage, layer=benchmark_route.layer_name_project_area
                )
                .iloc[0]
                .geometry,
                human_designed_route=human_designed_route,
                start_mid_end_points=route_stops,
                raster_name_prefix=benchmark_route.raster_name_prefix,
                compute_rasters_in_parallel=True,
                compute_metrics=True,
            )


if __name__ == "__main__":
    app()
