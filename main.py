# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0
import pathlib
import time

import shapely
import structlog

from utility_route_planner.models.benchmark_routes import BenchmarkRouteCollection
from utility_route_planner.models.lcpa.lcpa_engine import LcpaUtilityRouteEngine
from settings import Config
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.route_evaluation_metrics import RouteEvaluationMetrics
from utility_route_planner.util.geo_utilities import get_first_last_point_from_linestring
from utility_route_planner.util.write import reset_geopackage
import geopandas as gpd

logger = structlog.get_logger(__name__)


def run_mcda_lcpa(
    preset: str,
    path_geopackage_mcda_input: pathlib.Path,
    project_area_geometry: shapely.Polygon,
    start_mid_end_points: tuple,
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


if __name__ == "__main__":
    reset_geopackage(Config.PATH_GEOPACKAGE_LCPA_OUTPUT, truncate=True)
    route_collection = BenchmarkRouteCollection()

    cases_to_run = [1, 2, 3, 4, 5]  # 1 to 5, or [] for all
    for case in route_collection.get_routes(cases_to_run):
        human_designed_route = (
            gpd.read_file(case.path_geopackage, layer=case.layer_name_human_designed_route).iloc[0].geometry
        )
        route_stops = get_first_last_point_from_linestring(human_designed_route)
        if case.stops:
            route_stops = list(route_stops)[:1] + [shapely.Point(i) for i in case.stops] + list(route_stops)[1:]  # type: ignore

        run_mcda_lcpa(
            "preset_benchmark_raw",
            case.path_geopackage,
            gpd.read_file(case.path_geopackage, layer=case.layer_name_project_area).iloc[0].geometry,
            route_stops,
            human_designed_route,
            case.raster_name_prefix,
            compute_rasters_in_parallel=True,
        )
