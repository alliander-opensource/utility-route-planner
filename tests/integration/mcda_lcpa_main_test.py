# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import shapely

from main import run_mcda_lcpa
from settings import Config
from utility_route_planner.models.lcpa.lcpa_engine import LcpaUtilityRouteEngine
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.util.geo_utilities import get_first_last_point_from_linestring
from utility_route_planner.util.write import write_results_to_geopackage
import geopandas as gpd


@pytest.mark.usefixtures("setup_mcda_lcpa_testing")
class TestMcdaLcpaChain:
    @pytest.mark.parametrize(
        "utility_route_sketch",
        (
            [(174896.9, 451130.5), (175279.7, 450519.6)],
            [(174896.9, 451130.5), (174968.1, 450985.7), (174975.1, 450731.1), (175279.7, 450519.6)],
        ),
    )
    def test_mcda_lcpa_chain_pytest_files(self, utility_route_sketch):
        mcda_engine = McdaCostSurfaceEngine(
            "preset_benchmark_raw",
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry,
        )
        mcda_engine.preprocess_vectors()
        path_suitability_raster = mcda_engine.preprocess_rasters(
            mcda_engine.processed_vectors,
            cell_size=0.5,
            max_block_size=2048,
            run_in_parallel=False,
        )

        lcpa_engine = LcpaUtilityRouteEngine()
        lcpa_engine.get_lcpa_route(
            path_suitability_raster,
            shapely.LineString(utility_route_sketch),
            mcda_engine.raster_preset.general.project_area_geometry,
        )
        write_results_to_geopackage(Config.PATH_GEOPACKAGE_LCPA_OUTPUT, lcpa_engine.lcpa_result, "utility_route_result")


@pytest.mark.parametrize(
    "path_geopackage, layer_name_project_area, layer_name_utility_route_human_designed",
    [
        (
            Config.PATH_GEOPACKAGE_CASE_01,
            Config.LAYER_NAME_PROJECT_AREA_CASE_01,
            Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_01,
        ),
        (
            Config.PATH_GEOPACKAGE_CASE_02,
            Config.LAYER_NAME_PROJECT_AREA_CASE_02,
            Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_02,
        ),
        (
            Config.PATH_GEOPACKAGE_CASE_03,
            Config.LAYER_NAME_PROJECT_AREA_CASE_03,
            Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_03,
        ),
        (
            Config.PATH_GEOPACKAGE_CASE_04,
            Config.LAYER_NAME_PROJECT_AREA_CASE_04,
            Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_04,
        ),
        (
            Config.PATH_GEOPACKAGE_CASE_05,
            Config.LAYER_NAME_PROJECT_AREA_CASE_05,
            Config.LAYER_NAME_HUMAN_DESIGNED_ROUTE_CASE_05,
        ),
    ],
)
def test_mcda_lcpa_chain_all_benchmark_cases(
    path_geopackage, layer_name_project_area, layer_name_utility_route_human_designed
):
    human_designed_route = (
        gpd.read_file(path_geopackage, layer=layer_name_utility_route_human_designed).iloc[0].geometry
    )
    start_end_point = get_first_last_point_from_linestring(human_designed_route)

    run_mcda_lcpa(
        Config.RASTER_PRESET_NAME_BENCHMARK,
        path_geopackage,
        gpd.read_file(path_geopackage, layer=layer_name_project_area).geometry.iloc[0],
        start_end_point,
        human_designed_route,
        raster_name_prefix="",
        compute_rasters_in_parallel=False,
    )
