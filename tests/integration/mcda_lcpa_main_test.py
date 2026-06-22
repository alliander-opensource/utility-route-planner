# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import shapely

from settings import Config
from utility_route_planner.models.lcpa.lcpa_engine import LcpaUtilityRouteEngine
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
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
