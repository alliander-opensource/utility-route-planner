import geopandas as gpd
import pytest
import shapely
from shapely.geometry.linestring import LineString

from utility_route_planner.models.lcpa.lcpa_engine import LcpaUtilityRouteEngine
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from settings import Config
from utility_route_planner.util.write import write_to_file


class TestVRTRouting:
    @pytest.fixture(scope="session")
    def preprocessed_vectors(self) -> McdaCostSurfaceEngine:
        mcda_engine = McdaCostSurfaceEngine(
            Config.RASTER_PRESET_NAME_BENCHMARK,
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            gpd.read_file(
                Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA
            ).geometry.iloc[0],
        )
        mcda_engine.preprocess_vectors()

        return mcda_engine

    @pytest.fixture(scope="session")
    def start_end_point_route(self) -> tuple[shapely.Point, shapely.Point]:
        start_point = shapely.Point(174823.0, 450979.7)
        end_point = shapely.Point(175841.0, 450424.2)

        return start_point, end_point

    @pytest.fixture(scope="session")
    def route_for_single_tiff_file(
        self,
        preprocessed_vectors: McdaCostSurfaceEngine,
        start_end_point_route,
    ) -> shapely.LineString:
        mcda_engine = preprocessed_vectors

        # Use a max block size that exceeds the width & height of the project area (to force using a single tiff)
        max_block_size = 2048
        raster_path_vrt = mcda_engine.preprocess_rasters(
            mcda_engine.processed_vectors, cell_size=0.5, max_block_size=max_block_size, run_in_parallel=False
        )

        lcpa_engine = LcpaUtilityRouteEngine()
        lcpa_route = lcpa_engine.get_lcpa_route(
            raster_path_vrt,
            shapely.LineString(start_end_point_route),
            mcda_engine.raster_preset.general.project_area_geometry,
        )
        return lcpa_route

    @pytest.mark.parametrize("max_block_size", [256, 512, 1024])
    @pytest.mark.usefixtures("setup_mcda_lcpa_testing")
    def test_vrt_results_in_same_route_as_single_tiff(
        self,
        preprocessed_vectors: McdaCostSurfaceEngine,
        start_end_point_route: tuple[shapely.Point, shapely.Point],
        route_for_single_tiff_file: LineString,
        max_block_size: int,
        debug: bool = False,
    ):
        # Given the set of preprocessed vectors, preprocess the rasters given the max block size
        mcda_engine = preprocessed_vectors
        raster_path_vrt = mcda_engine.preprocess_rasters(
            mcda_engine.processed_vectors, cell_size=0.5, max_block_size=max_block_size, run_in_parallel=False
        )

        lcpa_engine = LcpaUtilityRouteEngine()
        route_using_vrt = lcpa_engine.get_lcpa_route(
            raster_path_vrt,
            shapely.LineString(start_end_point_route),
            mcda_engine.raster_preset.general.project_area_geometry,
        )

        # Verify that the route that was computed using the vrt raster is equal to the route that was computed using
        # the standard tiff file.
        assert route_using_vrt.equals(route_for_single_tiff_file)

        if debug:
            write_to_file(route_using_vrt, "route_using_vrt.fgb")
            write_to_file(route_for_single_tiff_file, "route_using_tif.fgb")
