# SPDX-FileCopyrightText: Contributors to the utility-route-project and Alliander N.V.
#
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as et

import pytest
import geopandas as gpd
import rasterio
import rasterio.sample
import shapely
import numpy as np
from rasterio.features import geometry_mask
from rasterio.transform import rowcol

from utility_route_planner.models.mcda.mcda_datastructures import RasterizedCriterion
from settings import Config
from utility_route_planner.models.mcda.exceptions import (
    RasterCellSizeTooSmall,
    InvalidSuitabilityRasterInput,
    InvalidGroupValue,
)
from utility_route_planner.models.mcda.mcda_engine import McdaCostSurfaceEngine
from utility_route_planner.models.mcda.mcda_presets import preset_collection
from utility_route_planner.models.mcda.mcda_rasterizing import (
    rasterize_vector_data,
    merge_criteria_rasters,
    get_raster_settings,
    write_raster_block,
    clip_raster_mask_to_project_area,
)
from utility_route_planner.util.write import reset_geopackage


@pytest.fixture
def setup_clean_start():
    reset_geopackage(Config.PATH_GEOPACKAGE_MCDA_OUTPUT, truncate=False)


@pytest.mark.usefixtures("setup_clean_start")
class TestRasterPreprocessing:
    def test_preprocess_single_raster(self):
        preset_to_load = {
            "general": preset_collection["preset_benchmark_raw"]["general"],
            "criteria": {
                "small_above_ground_obstacles": preset_collection["preset_benchmark_raw"]["criteria"][
                    "small_above_ground_obstacles"
                ],
            },
        }
        mcda_engine = McdaCostSurfaceEngine(
            preset_to_load,
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry,
        )
        mcda_engine.preprocess_vectors()
        mcda_engine.preprocess_rasters(
            mcda_engine.processed_vectors,
            cell_size=0.5,
            max_block_size=2048,
            run_in_parallel=False,
        )
        assert mcda_engine.processed_criteria_names == {"small_above_ground_obstacles"}
        assert mcda_engine.unprocessed_criteria_names == set()

    def test_preprocess_all_rasters(self):
        mcda_engine = McdaCostSurfaceEngine(
            Config.RASTER_PRESET_NAME_BENCHMARK,
            Config.PYTEST_PATH_GEOPACKAGE_MCDA,
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry,
        )
        mcda_engine.preprocess_vectors()
        mcda_engine.preprocess_rasters(
            mcda_engine.processed_vectors,
            cell_size=0.5,
            max_block_size=2048,
            run_in_parallel=False,
        )
        assert mcda_engine.processed_criteria_names == {
            "begroeid_terreindeel",
            "waterdeel",
            "ondersteunend_wegdeel",
            "pand",
            "wegdeel",
            "excluded_area",
            "onbegroeid_terreindeel",
            "vegetation_object",
            "small_above_ground_obstacles",
        }
        assert mcda_engine.unprocessed_criteria_names == {
            "ondersteunend_waterdeel",
            "overig_bouwwerk",
            "kunstwerkdeel",
            "protected_area",
            "existing_utilities",
            "existing_substations",
        }

    def test_preprocess_all_rasters_correct_in_vrt_file(self):
        mcda_engine = McdaCostSurfaceEngine(
            Config.RASTER_PRESET_NAME_BENCHMARK,
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

        # Verify that the raster can be opened by rasterio
        band_nr = 1
        with rasterio.open(path_suitability_raster, "r") as src:
            src.read(band_nr)
            raster_meta_data = src.meta

        # Verify the CRS is correct
        assert raster_meta_data["crs"] == Config.CRS

        # Verify size of raster is equal to size of project area grid
        x_min, y_min, x_max, y_max = mcda_engine.project_area_grid.total_bounds
        assert (x_max - x_min) / Config.RASTER_CELL_SIZE == raster_meta_data["width"]
        assert (y_max - y_min) / Config.RASTER_CELL_SIZE == raster_meta_data["height"]

        # Verify that all raster blocks are present in the VRT file
        vrt_tree = et.parse(path_suitability_raster)
        bands = vrt_tree.getroot().find("VRTRasterBand")
        sources = bands.findall("ComplexSource")
        assert len(sources) == len(mcda_engine.project_area_grid)


def test_rasterize_vector_data_cell_size_error():
    with pytest.raises(RasterCellSizeTooSmall):
        project_area = (
            gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA)
            .iloc[0]
            .geometry
        )
        get_raster_settings(project_area, cell_size=500000)


def test_rasterize_single_criterion(debug=False):
    max_value = Config.INTERMEDIATE_RASTER_VALUE_LIMIT_UPPER
    min_value = Config.INTERMEDIATE_RASTER_VALUE_LIMIT_LOWER
    no_data = Config.INTERMEDIATE_RASTER_NO_DATA

    gdf = gpd.GeoDataFrame(
        data=[
            # These layers all overlap each other.
            [1, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
            [10, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
            [10, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
            [1, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
            # One larger partly overlapping polygon with a unique value.
            [5, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]]).buffer(50)],
            # These values should be reset to the min/max of the intermediate raster values
            [
                min_value - 1000,
                shapely.Polygon([[175091, 450919], [175091, 450911], [175105, 450911], [175091, 450919]]),
            ],
            [
                max_value + 1000,
                shapely.Polygon([[175012, 450920], [175011, 450907], [175019, 450906], [175012, 450920]]),
            ],
            # This value is equal to no-data and should be reset to a "safe" value (+1 it)
            [no_data, shapely.Polygon([[174917, 450965], [174937, 450962], [174916, 450952], [174917, 450965]])],
        ],
        geometry="geometry",
        crs=Config.CRS,
        columns=["suitability_value", "geometry"],
    )
    points_to_sample = gpd.GeoDataFrame(
        data=[
            [1, 10, shapely.Point(174872.396, 451084.460)],
            [2, 5, shapely.Point(174868.573, 451086.020)],
            [3, no_data, shapely.Point(174985.83, 451101.57)],
            [4, no_data, shapely.Point(174686.5, 451164.9)],
            [5, max_value, shapely.Point(175013, 450909)],
            [6, min_value, shapely.Point(175094, 450913)],
            [7, no_data + 1, shapely.Point(174923.49, 450959.17)],
        ],
        geometry="geometry",
        crs=Config.CRS,
        columns=["sample_id", "expected_suitability_value", "geometry"],
    )
    if debug:
        # Using QGIS, it is easier doublecheck what values we are expecting in this test.
        gdf.to_file(Config.PATH_RESULTS / "pytest_rasterize_single_criterion.geojson")
        points_to_sample.to_file(Config.PATH_RESULTS / "pytest_rasterize_single_criterion_points_to_sample.geojson")

    # The order of the values should not matter, check this.
    sort_asc = gdf.sort_values("suitability_value", ascending=True).copy()
    sort_desc = gdf.sort_values("suitability_value", ascending=False).copy()

    gdfs_to_rasterize = [gdf, sort_desc, sort_asc]
    project_area = (
        gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA).iloc[0].geometry
    )
    raster_settings = get_raster_settings(project_area, 0.5)
    for gdf in gdfs_to_rasterize:
        rasterized_vector = rasterize_vector_data("test_rasterize", gdf, raster_settings)
        unique_values = np.unique(rasterized_vector)
        assert set(unique_values) == {no_data, min_value, 5, 10, max_value}
        # Check that the overlapping part has the highest value
        for _, row in points_to_sample.iterrows():
            row_index, col_index = rowcol(raster_settings.transform, row.geometry.x, row.geometry.y)
            assert rasterized_vector[int(row_index)][int(col_index)] == row.expected_suitability_value


def test_sum_rasters(debug=False):
    max_value = Config.FINAL_RASTER_VALUE_LIMIT_UPPER
    min_value = Config.FINAL_RASTER_VALUE_LIMIT_LOWER
    no_data = Config.FINAL_RASTER_NO_DATA
    # 4 rasters:
    # 1. group a - partial overlap
    criterion_a_1 = gpd.GeoDataFrame(
        data=[
            # These layers all overlap each other.
            [1, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
            [10, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]])],
            # One larger partly overlapping polygon with a unique value.
            [5, shapely.Polygon([[174872, 451093], [174870, 451082], [174876, 451081], [174872, 451093]]).buffer(50)],
            # These values should be reset to the min/max of the intermediate raster values
            [
                min_value - 1000,
                shapely.Polygon([[175091, 450919], [175091, 450911], [175105, 450911], [175091, 450919]]),
            ],
            [
                max_value + 1000,
                shapely.Polygon([[175012, 450920], [175011, 450907], [175019, 450906], [175012, 450920]]),
            ],
        ],
        geometry="geometry",
        crs=Config.CRS,
        columns=["suitability_value", "geometry"],
    )
    # 2. group a - partial overlap
    criterion_a_2 = gpd.GeoDataFrame(
        data=[
            # Overlaps criterion a 1 with a higher value
            [50, shapely.Polygon([[174797, 451107], [174944, 451090], [174807, 451129], [174797, 451107]])],
        ],
        geometry="geometry",
        crs=Config.CRS,
        columns=["suitability_value", "geometry"],
    )
    # 3. group b - partial overlap
    criterion_b_1 = gpd.GeoDataFrame(
        data=[
            # Overlaps criterion a 1 with a higher value
            [20, shapely.Point([174813.28, 451113.88])],
            [1000, shapely.Point([174870.46, 451051.07])],
            [-20, shapely.Point([175013.310, 450910.294])],
            [-1, shapely.Polygon([[175087, 450911], [175107, 450912], [175087, 450915], [175087, 450911]])],
            # Overlaps criterion a 1 with the same value but signed.
            [-5, shapely.Polygon([[174830, 451074], [174842, 451069], [174831, 451061], [174830, 451074]])],
        ],
        geometry="geometry",
        crs=Config.CRS,
        columns=["suitability_value", "geometry"],
    )
    # 4. group b - overlaps criterion b1 and a1
    criterion_b_2 = gpd.GeoDataFrame(
        data=[
            [15, shapely.Polygon([[175096, 450908], [175089, 450908], [175091, 450921], [175096, 450908]])],
        ],
        geometry="geometry",
        crs=Config.CRS,
        columns=["suitability_value", "geometry"],
    )
    # 5. group b - tree on the edge of the project area. The buffer exceeds the project area boundary.
    criterion_b_3 = gpd.GeoDataFrame(
        data=[[10, shapely.Point(174741.950, 451113.084).buffer(10)]],
        geometry="geometry",
        crs=Config.CRS,
        columns=["suitability_value", "geometry"],
    )
    # 6. group c - overlapping a1
    criterion_c_1 = gpd.GeoDataFrame(
        data=[
            [1, shapely.Polygon([[174729, 451158], [174940, 451115], [174841, 451195], [174729, 451158]])],
            [10, shapely.Polygon([[174915, 451128], [174924, 451135], [174926, 451109], [174915, 451128]])],
        ],
        geometry="geometry",
        crs=Config.CRS,
        columns=["suitability_value", "geometry"],
    )
    # 7. group c - overlapping b1 and c1
    criterion_c_2 = gpd.GeoDataFrame(
        data=[
            [1, shapely.Polygon([[175090, 450906], [175103, 450905], [175096, 450918], [175090, 450906]])],
            [39, shapely.Polygon([[174811, 451226], [174834, 451155], [174909, 451174], [174811, 451226]])],
        ],
        geometry="geometry",
        crs=Config.CRS,
        columns=["suitability_value", "geometry"],
    )
    points_to_sample = gpd.GeoDataFrame(
        data=[
            [1, 14, shapely.Point(175090.35, 450911.67)],  # overlap between b1 and b2
            [2, min_value, shapely.Point(175091.8234, 450911.7488)],  # overlap between a1, b1 and b2
            [3, min_value, shapely.Point(175088.2180, 450912.7950)],  # only b1
            [4, Config.FINAL_RASTER_NO_DATA, shapely.Point(174249.6, 451268.1)],  # out of extent and project area
            [5, max_value, shapely.Point(175013.3110, 450910.3013)],  # overlap between b1 and a1
            [6, 5, shapely.Point(174839.089, 451050.785)],  # just a1
            [7, 70, shapely.Point(174813.2646, 451113.9146)],  # just a2
            [
                8,
                Config.FINAL_RASTER_NO_DATA,
                shapely.Point(174763.32, 451347.41),
            ],  # out of the project area, within extent
            [9, 1, shapely.Point(174833.90, 451067.57)],  # b1 and a1 sum is 0 here, reset to a valid value of 1.
            [10, Config.FINAL_RASTER_NO_DATA, shapely.Point(174878.65, 451132.89)],  # c1 overlaps a1
            [11, Config.FINAL_RASTER_NO_DATA, shapely.Point(174799.54, 451170.54)],  # c1
            [12, Config.FINAL_RASTER_NO_DATA, shapely.Point(174921.44, 451123.59)],  # c1 overlapping c1
            [13, Config.FINAL_RASTER_NO_DATA, shapely.Point(174745.32, 451159.41)],  # c1 outside the project area
            [14, Config.FINAL_RASTER_NO_DATA, shapely.Point(175092.267, 450908.932)],  # c2 overlapping b2
            [15, Config.FINAL_RASTER_NO_DATA, shapely.Point(175097.673, 450912.390)],  # c2 overlapping b2, a1
            [16, Config.FINAL_RASTER_NO_DATA, shapely.Point(174847.32, 451177.96)],  # c2 overlapping c1
        ],
        geometry="geometry",
        crs=Config.CRS,
        columns=["sample_id", "expected_suitability_value", "geometry"],
    )
    if debug:
        # Using QGIS, it is easier doublecheck what values we are expecting in this test.
        criterion_a_1.to_file(Config.PATH_RESULTS / "pytest_sum_criterion_a1.geojson")
        criterion_a_2.to_file(Config.PATH_RESULTS / "pytest_sum_criterion_a2.geojson")
        criterion_b_1.to_file(Config.PATH_RESULTS / "pytest_sum_criterion_b1.geojson")
        criterion_b_2.to_file(Config.PATH_RESULTS / "pytest_sum_criterion_b2.geojson")
        criterion_b_3.to_file(Config.PATH_RESULTS / "pytest_sum_criterion_b3.geojson")
        criterion_c_1.to_file(Config.PATH_RESULTS / "pytest_sum_criterion_c1.geojson")
        criterion_c_2.to_file(Config.PATH_RESULTS / "pytest_sum_criterion_c2.geojson")
        points_to_sample.to_file(Config.PATH_RESULTS / "pytest_sum_points_to_sample.geojson")

    rasters_to_merge = []
    project_area = (
        gpd.read_file(Config.PYTEST_PATH_GEOPACKAGE_MCDA, layer=Config.PYTEST_LAYER_NAME_PROJECT_AREA).iloc[0].geometry
    )
    raster_settings = get_raster_settings(project_area, 0.5)
    for (
        group,
        criterion_gdf,
        criterion_name,
    ) in [
        ["a", criterion_a_1, "criterion_a1"],
        ["a", criterion_a_2, "criterion_a2"],
        ["b", criterion_b_1, "criterion_b1"],
        ["b", criterion_b_2, "criterion_b2"],
        ["b", criterion_b_3, "criterion_b3"],
        ["c", criterion_c_1, "criterion_c1"],
        ["c", criterion_c_2, "criterion_c2"],
    ]:
        rasterized_vector = rasterize_vector_data(criterion_name, criterion_gdf, raster_settings)
        rasters_to_merge.append(RasterizedCriterion(criterion_name, rasterized_vector, group))

    merged_raster = merge_criteria_rasters(rasters_to_merge, raster_settings.height, raster_settings.width)
    merged_raster = clip_raster_mask_to_project_area(merged_raster, project_area, raster_settings.transform)

    path_suitability_raster, _ = write_raster_block(merged_raster, raster_settings, "pytest_suitability_raster")
    with rasterio.open(path_suitability_raster, "r") as out:
        result = out.read(1)
        unique_values = np.unique(result)
        assert set(unique_values) == {no_data, min_value, 5, 10, 14, 15, 50, 70, max_value}
        # Check that the overlapping part has the highest value
        for _, row in points_to_sample.iterrows():
            values = list(rasterio.sample.sample_gen(out, [[row.geometry.x, row.geometry.y]]))
            assert values[0][0] == row.expected_suitability_value

        # Verify that all datapoints outside the project area are set to no data
        mask = geometry_mask([project_area], transform=out.transform, out_shape=out.shape)
        assert np.array_equal(np.unique(result[np.where(mask)]), [no_data])


@pytest.mark.parametrize(
    "invalid_input",
    [
        [RasterizedCriterion("c1", np.array([]), "d")],
        [RasterizedCriterion("c2", np.array([]), "f")],
        [RasterizedCriterion("c2", np.array([]), "e"), RasterizedCriterion("c2", np.array([]), "d")],
    ],
)
def test_invalid_group_value_in_suitability_raster(invalid_input):
    with pytest.raises(InvalidGroupValue):
        merge_criteria_rasters(invalid_input, 10, 10)


def test_invalid_suitability_raster_input():
    with pytest.raises(InvalidSuitabilityRasterInput):
        merge_criteria_rasters([], 10, 10)
